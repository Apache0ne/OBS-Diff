"""Tensor-only export/load for physically structured ConvRot SDXL UNets.

SafeTensors cannot store the Python UNet object carried by the original .pth.
This module stores explicit SDXL config/shape metadata plus packed Comfy-Kitchen
qdata/scales and reconstructs the exact pruned UNet in code. It can also replace
the old UNet in a full SDXL SafeTensors file while preserving both INT4 CLIPs
and the FP16 VAE.
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter, OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from .convrot_sdxl import load_comfy_quant_runtime, model_actual_storage_bytes, quantized_layout_counts

SCHEMA = "apacheone.structured_convrot_full_sdxl.v1"
ROOT = "__structured_convrot_unet__"
SOURCE_UNET_PREFIX = "model.diffusion_model."
CLIP_PREFIXES = ("conditioner.embedders.0.transformer.", "conditioner.embedders.1.model.")
VAE_PREFIX = "first_stage_model."

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
DTYPE_NAMES = {value: key for key, value in DTYPES.items()}


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype not in DTYPE_NAMES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return DTYPE_NAMES[dtype]


def _resolve_parent(root: nn.Module, name: str):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part] if part in parent._modules else getattr(parent, part)
    return parent, parts[-1]


def _module(root: nn.Module, name: str):
    value = root
    for part in name.split("."):
        value = value._modules[part] if part in value._modules else getattr(value, part)
    return value


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous().clone()


def _encode_param_value(value: Any, tensors: MutableMapping[str, torch.Tensor], key: str):
    if isinstance(value, torch.Tensor):
        tensors[key] = _cpu(value)
        return {"tensor": key}
    if isinstance(value, torch.dtype):
        return {"dtype": _dtype_name(value)}
    if isinstance(value, tuple):
        return {"tuple": list(value)}
    return value


def _decode_param_value(value: Any, sf):
    if isinstance(value, Mapping) and "tensor" in value:
        return sf.get_tensor(str(value["tensor"])).contiguous()
    if isinstance(value, Mapping) and "dtype" in value:
        return DTYPES[str(value["dtype"])]
    if isinstance(value, Mapping) and "tuple" in value:
        return tuple(value["tuple"])
    return value


def _architecture(unet: nn.Module):
    config = unet.config.to_dict()
    linears, attentions = [], []
    for path, module in unet.named_modules(remove_duplicate=False):
        if isinstance(module, nn.Linear):
            linears.append({
                "path": path,
                "in": int(module.in_features),
                "out": int(module.out_features),
                "bias": module.bias is not None,
                "dtype": _dtype_name(module.weight.dtype),
            })
        if all(hasattr(module, x) for x in ("heads", "inner_dim", "to_q", "to_k", "to_v", "to_out")):
            attentions.append({
                "path": path,
                "heads": int(module.heads),
                "inner_dim": int(module.inner_dim),
                "inner_kv_dim": int(getattr(module, "inner_kv_dim", module.inner_dim)),
                "sliceable_head_dim": int(getattr(module, "sliceable_head_dim", module.heads)),
            })
    return {"config": config, "linears": linears, "attentions": attentions}


def _unet_payload(unet: nn.Module):
    tensors = OrderedDict()
    normal, quantized, buffers = {}, {}, {}
    for name, parameter in unet.named_parameters(remove_duplicate=False):
        if hasattr(parameter, "_qdata") and hasattr(parameter, "_params"):
            base = f"{ROOT}.quant.{name}"
            qkey = base + ".qdata"
            tensors[qkey] = _cpu(parameter._qdata)
            params = {}
            for field in dataclasses.fields(parameter._params):
                params[field.name] = _encode_param_value(
                    getattr(parameter._params, field.name), tensors, base + ".param." + field.name
                )
            quantized[name] = {
                "layout": str(parameter._layout_cls),
                "qdata": qkey,
                "params": params,
                "requires_grad": bool(parameter.requires_grad),
            }
        else:
            key = f"{ROOT}.param.{name}"
            tensors[key] = _cpu(parameter)
            normal[name] = {"tensor": key, "requires_grad": bool(parameter.requires_grad)}
    for name, buffer in unet.named_buffers(remove_duplicate=False):
        key = f"{ROOT}.buffer.{name}"
        tensors[key] = _cpu(buffer)
        buffers[name] = key
    return tensors, {
        "normal": normal,
        "quantized": quantized,
        "buffers": buffers,
        "logical_parameters": int(sum(x.numel() for x in unet.parameters())),
        "storage_bytes": int(model_actual_storage_bytes(unet)),
        "layout_counts": dict(quantized_layout_counts(unet)),
    }


def _text_manifest(metadata: Mapping[str, str]):
    for key in ("structured_convrot_full_manifest", "single_backend_manifest", "hybrid_quant_manifest"):
        raw = metadata.get(key)
        if not raw:
            continue
        try:
            text = json.loads(raw).get("text_encoders")
            if isinstance(text, Mapping):
                return dict(text)
        except Exception:
            pass
    return {}


def _copy_source_without_unet(source: Path, output):
    counts = Counter()
    with safe_open(str(source), framework="pt", device="cpu") as sf:
        metadata = dict(sf.metadata() or {})
        for key in sf.keys():
            if key.startswith(SOURCE_UNET_PREFIX):
                counts["removed_old_unet"] += 1
                continue
            output[key] = sf.get_tensor(key).contiguous()
            counts["clip" if key.startswith(CLIP_PREFIXES) else "vae" if key.startswith(VAE_PREFIX) else "other"] += 1
    return metadata, dict(counts)


def build_full_checkpoint(
    quantized_unet_pth: str | Path,
    clip_vae_source_safetensors: str | Path,
    output_path: str | Path,
    *,
    profile: str | None = None,
    overwrite: bool = False,
):
    load_comfy_quant_runtime()
    pth = Path(quantized_unet_pth).resolve()
    source = Path(clip_vae_source_safetensors).resolve()
    output_path = Path(output_path).resolve()
    if not pth.is_file():
        raise FileNotFoundError(pth)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unet = torch.load(str(pth), map_location="cpu", weights_only=False)
    if not isinstance(unet, nn.Module):
        raise TypeError(f"Expected UNet nn.Module, got {type(unet)}")
    unet.eval()
    layouts = dict(quantized_layout_counts(unet))
    if not layouts:
        raise RuntimeError("UNet contains no Comfy-Kitchen quantized layouts")

    output = OrderedDict()
    source_metadata, copy_counts = _copy_source_without_unet(source, output)
    unet_tensors, tensor_manifest = _unet_payload(unet)
    output.update(unet_tensors)
    text = _text_manifest(source_metadata)
    if int(text.get("count", 0)) != 200:
        raise RuntimeError(f"Expected 200 INT4 CLIP matrices, got {text.get('count')}")

    profile = profile or pth.parent.name
    manifest = {
        "schema": SCHEMA,
        "profile": profile,
        "source_unet_pth": str(pth),
        "source_clip_vae": str(source),
        "unet": {"architecture": _architecture(unet), "tensors": tensor_manifest},
        "text_encoders": text,
        "vae": {"prefix": VAE_PREFIX, "dtype": "float16"},
        "copy_counts": copy_counts,
        "runtime": {
            "loader": "lib.structured_convrot_full_safetensors.load_structured_convrot_unet",
            "dense_unet_required": False,
            "persistent_dequantization": False,
        },
    }
    metadata = dict(source_metadata)
    metadata.pop("single_backend_manifest", None)
    metadata.pop("hybrid_quant_manifest", None)
    metadata.update({
        "format": "pt",
        "quantization": "structured_obs_diff_comfy_convrot_clip_int4",
        "structured_convrot_schema": SCHEMA,
        "structured_convrot_profile": profile,
        "structured_convrot_full_manifest": json.dumps(manifest, separators=(",", ":"), sort_keys=True),
        "unet_backend": profile,
        "clip_backend": "convrot_w4a4_int4",
        "clip_convrot_int4_layers": "200",
        "vae_dtype": "float16",
        "custom_loader_required": "true",
        "dense_unet_required": "false",
    })
    if output_path.exists():
        output_path.unlink()
    save_file(output, str(output_path), metadata=metadata)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "tensor_count": len(output),
        "profile": profile,
        "unet_parameters": tensor_manifest["logical_parameters"],
        "unet_storage_bytes": tensor_manifest["storage_bytes"],
        "unet_layout_counts": layouts,
        "clip_target_count": 200,
        "copy_counts": copy_counts,
    }


def full_manifest(path: str | Path):
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        metadata = dict(sf.metadata() or {})
    raw = metadata.get("structured_convrot_full_manifest")
    if not raw:
        raise ValueError("Missing structured_convrot_full_manifest")
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported schema: {manifest.get('schema')}")
    return manifest


def _empty_context():
    try:
        from accelerate import init_empty_weights
        return init_empty_weights(include_buffers=True)
    except Exception:
        return nullcontext()


def _replace_linears(unet: nn.Module, records: Sequence[Mapping[str, Any]]):
    for record in records:
        parent, leaf = _resolve_parent(unet, str(record["path"]))
        old = parent._modules.get(leaf)
        if not isinstance(old, nn.Linear):
            raise TypeError(f"Expected Linear at {record['path']}, got {type(old)}")
        parent._modules[leaf] = nn.Linear(
            int(record["in"]), int(record["out"]), bias=bool(record["bias"]),
            device="meta", dtype=DTYPES[str(record["dtype"])],
        )


def _set_parameter(root, name, tensor, requires_grad=False):
    parent, leaf = _resolve_parent(root, name)
    parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=requires_grad)


def _set_buffer(root, name, tensor):
    parent, leaf = _resolve_parent(root, name)
    parent._buffers[leaf] = tensor


def load_structured_convrot_unet(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
):
    """Load the structured packed UNet from SafeTensors without a dense UNet."""
    load_comfy_quant_runtime()
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.base import get_layout_class
    from diffusers import UNet2DConditionModel

    manifest = full_manifest(path)
    unet_record = manifest["unet"]
    architecture = unet_record["architecture"]
    tensor_manifest = unet_record["tensors"]
    with _empty_context():
        unet = UNet2DConditionModel.from_config(architecture["config"])
    _replace_linears(unet, architecture["linears"])
    for record in architecture["attentions"]:
        module = _module(unet, str(record["path"]))
        module.heads = int(record["heads"])
        module.inner_dim = int(record["inner_dim"])
        module.inner_kv_dim = int(record["inner_kv_dim"])
        module.sliceable_head_dim = int(record["sliceable_head_dim"])
        if hasattr(module, "fused_projections"):
            module.fused_projections = False

    device = torch.device(device)
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        for name, record in tensor_manifest["normal"].items():
            tensor = sf.get_tensor(record["tensor"]).contiguous()
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            _set_parameter(unet, name, tensor.to(device), bool(record.get("requires_grad", False)))
        for name, record in tensor_manifest["quantized"].items():
            layout = str(record["layout"])
            layout_cls = get_layout_class(layout)
            params = {field: _decode_param_value(value, sf) for field, value in record["params"].items()}
            params = layout_cls.Params(**params).to_device(device)
            if dtype is not None:
                params = dataclasses.replace(params, orig_dtype=dtype)
            qdata = sf.get_tensor(record["qdata"]).contiguous().to(device)
            _set_parameter(
                unet, name, QuantizedTensor(qdata, layout, params), bool(record.get("requires_grad", False))
            )
        for name, key in tensor_manifest["buffers"].items():
            tensor = sf.get_tensor(key).contiguous()
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            _set_buffer(unet, name, tensor.to(device))

    meta = [name for name, p in unet.named_parameters() if p.device.type == "meta"]
    if meta:
        raise RuntimeError(f"Incomplete reconstruction; meta parameters remain: {meta[:5]}")
    actual = int(sum(x.numel() for x in unet.parameters()))
    expected = int(tensor_manifest["logical_parameters"])
    if actual != expected:
        raise RuntimeError(f"Parameter mismatch: {actual} != {expected}")
    layouts = dict(quantized_layout_counts(unet))
    if layouts != dict(tensor_manifest["layout_counts"]):
        raise RuntimeError(f"Layout mismatch: {layouts} != {tensor_manifest['layout_counts']}")
    unet.eval()
    return unet


def validate_full_checkpoint(path: str | Path, reconstruct_unet: bool = True):
    manifest = full_manifest(path)
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
    counts = {
        "clip_l": sum(key.startswith(CLIP_PREFIXES[0]) for key in keys),
        "clip_g": sum(key.startswith(CLIP_PREFIXES[1]) for key in keys),
        "vae": sum(key.startswith(VAE_PREFIX) for key in keys),
        "structured_unet": sum(key.startswith(ROOT) for key in keys),
    }
    if int(manifest.get("text_encoders", {}).get("count", 0)) != 200:
        raise RuntimeError("INT4 CLIP manifest is not 200 layers")
    if not counts["clip_l"] or not counts["clip_g"] or not counts["vae"] or not counts["structured_unet"]:
        raise RuntimeError(f"Missing checkpoint components: {counts}")
    result = {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "counts": counts}
    if reconstruct_unet:
        unet = load_structured_convrot_unet(path, device="cpu")
        result["unet_parameters"] = int(sum(x.numel() for x in unet.parameters()))
        result["unet_layout_counts"] = dict(quantized_layout_counts(unet))
        result["unet_storage_bytes"] = int(model_actual_storage_bytes(unet))
    return result

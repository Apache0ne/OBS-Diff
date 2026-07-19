#!/usr/bin/env python3
"""Comfy-Kitchen ConvRot quantization helpers for structured SDXL UNets.

The structured OBS-Diff UNet is a complete pickled Diffusers UNet with custom
per-block FFN widths and attention head counts. Standard ComfyUI SDXL model
construction cannot infer those shapes, so these helpers preserve the complete
UNet object and replace selected ``nn.Linear.weight`` tensors with the same
QuantizedTensor layouts used by current ComfyUI/comfy-kitchen:

* ``TensorWiseINT8Layout`` + ``convrot=True`` for W8A8 ConvRot.
* ``TensorCoreConvRotW4A4Layout`` for packed W4A4 ConvRot, including the
  ``linear_dtype`` selector added by ComfyUI PR #14859.

Convolutions, normalizations, embeddings and explicitly protected linears stay
FP16. The result is saved as another complete standalone UNet ``.pth`` object.
"""
from __future__ import annotations

import dataclasses
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn


COMFY_MIN_VERSION = (0, 2, 17)
INT8_GROUP_CANDIDATES = (256, 64)
W4A4_GROUP_SIZE = 256
W4A4_QUANT_GROUP_SIZE = 64


def _version_tuple(value: str) -> Tuple[int, ...]:
    output = []
    for part in value.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        output.append(int(digits))
    return tuple(output)


def load_comfy_quant_runtime():
    import comfy_kitchen
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout  # noqa: F401
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout  # noqa: F401

    version = getattr(comfy_kitchen, "__version__", "0.0.0")
    if _version_tuple(version) < COMFY_MIN_VERSION:
        raise RuntimeError(
            f"comfy-kitchen>={'.'.join(map(str, COMFY_MIN_VERSION))} is required; loaded {version}"
        )
    return comfy_kitchen, QuantizedTensor


def largest_int8_convrot_group(in_features: int) -> int | None:
    for group in INT8_GROUP_CANDIDATES:
        if in_features % group == 0:
            return group
    return None


def w4a4_eligible(module: nn.Linear) -> bool:
    return module.in_features % W4A4_GROUP_SIZE == 0 and module.out_features % 8 == 0


def is_transformer_linear(path: str) -> bool:
    return ".transformer_blocks." in f".{path}."


def is_cross_attention(path: str) -> bool:
    return ".attn2." in f".{path}."


def is_self_attention(path: str) -> bool:
    return ".attn1." in f".{path}."


def is_feed_forward(path: str) -> bool:
    return ".ff." in f".{path}."


def choose_quantization(path: str, module: nn.Linear, profile: str) -> Dict[str, Any]:
    """Return the Comfy quantization decision for one Linear.

    Profiles:
      convrot_int8
        W8A8 ConvRot for every transformer-block Linear. Other linears stay FP16.

      convrot_int4_mixed
        Quality-first PR #14859-style mixture: large FFN and self-attention
        linears use W4A4 when shape-compatible; all cross-attention and every
        unsupported pruned-width Linear use ConvRot INT8; outer SDXL linears
        remain FP16. Large FFNs use native INT4 MMA while smaller self-attention
        W4A4 layers select the INT8 MMA route.

      convrot_int4_fast
        W4A4 with native INT4 MMA for every compatible transformer Linear,
        with ConvRot INT8 fallback for incompatible pruned widths.
    """
    if not is_transformer_linear(path):
        return {"kind": "fp16", "reason": "protected_non_transformer_linear"}

    int8_group = largest_int8_convrot_group(module.in_features)
    if int8_group is None:
        return {"kind": "fp16", "reason": "no_supported_convrot_group"}

    if profile == "convrot_int8":
        return {"kind": "int8", "convrot_groupsize": int8_group}

    if profile == "convrot_int4_fast":
        if w4a4_eligible(module):
            return {
                "kind": "w4a4",
                "convrot_groupsize": W4A4_GROUP_SIZE,
                "quant_group_size": W4A4_QUANT_GROUP_SIZE,
                "linear_dtype": "int4",
            }
        return {
            "kind": "int8",
            "convrot_groupsize": int8_group,
            "reason": "w4a4_shape_fallback",
        }

    if profile == "convrot_int4_mixed":
        if is_cross_attention(path):
            return {
                "kind": "int8",
                "convrot_groupsize": int8_group,
                "reason": "cross_attention_quality_protection",
            }

        if w4a4_eligible(module) and (is_feed_forward(path) or is_self_attention(path)):
            linear_dtype = "int4" if is_feed_forward(path) else "int8"
            return {
                "kind": "w4a4",
                "convrot_groupsize": W4A4_GROUP_SIZE,
                "quant_group_size": W4A4_QUANT_GROUP_SIZE,
                "linear_dtype": linear_dtype,
            }

        return {
            "kind": "int8",
            "convrot_groupsize": int8_group,
            "reason": "mixed_profile_int8_fallback",
        }

    raise ValueError(f"Unknown ConvRot profile: {profile}")


def _param_tensor_bytes(params: Any) -> int:
    total = 0
    if dataclasses.is_dataclass(params):
        for field in dataclasses.fields(params):
            value = getattr(params, field.name)
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total


def tensor_actual_storage_bytes(value: torch.Tensor) -> int:
    if hasattr(value, "_qdata") and hasattr(value, "_params"):
        qdata = value._qdata
        return qdata.numel() * qdata.element_size() + _param_tensor_bytes(value._params)
    return value.numel() * value.element_size()


def model_actual_storage_bytes(model: nn.Module) -> int:
    seen = set()
    total = 0
    for _, parameter in model.named_parameters(remove_duplicate=False):
        identifier = id(parameter)
        if identifier in seen:
            continue
        seen.add(identifier)
        total += tensor_actual_storage_bytes(parameter)
    for _, buffer in model.named_buffers(remove_duplicate=False):
        identifier = id(buffer)
        if identifier in seen:
            continue
        seen.add(identifier)
        total += tensor_actual_storage_bytes(buffer)
    return total


def model_logical_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def quantized_layout_counts(model: nn.Module) -> Counter:
    counts = Counter()
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module.weight, "_layout_cls"):
            counts[str(module.weight._layout_cls)] += 1
    return counts


@torch.no_grad()
def quantize_linear_weight(
    module: nn.Linear,
    decision: Mapping[str, Any],
    *,
    quant_device: str,
    stochastic_rounding: int = 0,
) -> Dict[str, Any]:
    _, QuantizedTensor = load_comfy_quant_runtime()
    kind = str(decision["kind"])
    if kind == "fp16":
        return {
            "kind": "fp16",
            "logical_weight_values": module.weight.numel(),
            "actual_weight_bytes": module.weight.numel() * module.weight.element_size(),
        }

    source = module.weight.detach().to(device=quant_device, non_blocking=False).contiguous()
    original_dtype = source.dtype

    if kind == "int8":
        group = int(decision["convrot_groupsize"])
        quantized = QuantizedTensor.from_float(
            source,
            "TensorWiseINT8Layout",
            is_weight=True,
            per_channel=True,
            convrot=True,
            convrot_groupsize=group,
            stochastic_rounding=stochastic_rounding,
        )
    elif kind == "w4a4":
        quantized = QuantizedTensor.from_float(
            source,
            "TensorCoreConvRotW4A4Layout",
            convrot_groupsize=int(decision.get("convrot_groupsize", W4A4_GROUP_SIZE)),
            quant_group_size=int(decision.get("quant_group_size", W4A4_QUANT_GROUP_SIZE)),
            linear_dtype=str(decision.get("linear_dtype", "int4")),
            stochastic_rounding=stochastic_rounding,
        )
    else:
        raise ValueError(f"Unsupported decision kind: {kind}")

    quantized = quantized.to(device="cpu", dtype=original_dtype)
    module.weight = nn.Parameter(quantized, requires_grad=False)
    module.weight.requires_grad_(False)

    del source, quantized
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    weight = module.weight
    result = {
        "kind": kind,
        "logical_weight_values": int(weight.numel()),
        "actual_weight_bytes": int(tensor_actual_storage_bytes(weight)),
        "layout": str(getattr(weight, "_layout_cls", "")),
        "orig_shape": list(map(int, weight.shape)),
        "storage_shape": list(map(int, getattr(weight, "storage_shape", weight.shape))),
        "storage_dtype": str(getattr(weight, "storage_dtype", weight.dtype)),
    }
    if hasattr(weight, "_params"):
        params = weight._params
        for key in ("convrot_groupsize", "quant_group_size", "linear_dtype"):
            if hasattr(params, key):
                result[key] = getattr(params, key)
    return result


@torch.no_grad()
def quantize_unet(
    model: nn.Module,
    profile: str,
    *,
    quant_device: str = "cuda",
    stochastic_rounding: int = 0,
) -> Dict[str, Any]:
    if quant_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA quantization requested but CUDA is unavailable")

    model.eval()
    records = []
    logical_quantized_weights = 0
    actual_quantized_bytes = 0
    counts = Counter()

    linear_modules = [(path, module) for path, module in model.named_modules() if isinstance(module, nn.Linear)]
    for index, (path, module) in enumerate(linear_modules, 1):
        decision = choose_quantization(path, module, profile)
        before_values = int(module.weight.numel())
        before_bytes = int(module.weight.numel() * module.weight.element_size())
        info = quantize_linear_weight(
            module,
            decision,
            quant_device=quant_device,
            stochastic_rounding=stochastic_rounding,
        )
        counts[info["kind"]] += 1
        if info["kind"] != "fp16":
            logical_quantized_weights += before_values
            actual_quantized_bytes += int(info["actual_weight_bytes"])
        record = {
            "path": path,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "logical_weight_values": before_values,
            "original_fp16_bytes": before_bytes,
            **dict(decision),
            **info,
        }
        records.append(record)
        if index % 25 == 0 or index == len(linear_modules):
            print(
                f"  {profile}: linears {index}/{len(linear_modules)} | "
                f"fp16={counts['fp16']} int8={counts['int8']} w4a4={counts['w4a4']}"
            )
        gc.collect()

    total_parameters = model_logical_parameter_count(model)
    total_storage = model_actual_storage_bytes(model)
    return {
        "profile": profile,
        "linear_count": len(linear_modules),
        "decision_counts": dict(counts),
        "quantized_logical_weight_values": logical_quantized_weights,
        "quantized_fraction_of_unet_parameters": logical_quantized_weights / total_parameters,
        "quantized_weight_storage_bytes": actual_quantized_bytes,
        "actual_model_storage_bytes": total_storage,
        "logical_unet_parameters": total_parameters,
        "layout_counts": dict(quantized_layout_counts(model)),
        "layers": records,
    }


def save_quantized_unet(model: nn.Module, output_path: str | Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    model.eval()
    torch.save(model, output_path)
    manifest_path = output_path.parent / "convrot_manifest.json"
    payload = dict(manifest)
    payload.update(
        {
            "unet_path": str(output_path),
            "pth_bytes": output_path.stat().st_size,
            "load_example": (
                "unet = torch.load('unet_quantized.pth', map_location='cpu', "
                "weights_only=False)"
            ),
            "runtime_requirement": "comfy-kitchen>=0.2.17",
            "architecture_note": (
                "Complete structured SDXL UNet object; no dense UNet is needed. "
                "Standard ComfyUI SDXL construction cannot infer custom pruned shapes."
            ),
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def reload_verify(path: str | Path) -> nn.Module:
    load_comfy_quant_runtime()
    model = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module in {path}, got {type(model)}")
    model.eval()
    return model

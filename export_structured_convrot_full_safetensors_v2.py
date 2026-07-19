#!/usr/bin/env python3
"""Robust v2 exporter for structured ConvRot SDXL full SafeTensors.

Fixes two failure classes in the original exporter:
1. Source CLIP/VAE tensors are cloned while safe_open is still alive instead of
   retaining mmap-backed views after the context closes.
2. Architecture and layout metadata are normalized without losing tuple shape
   information required by Comfy-Kitchen reconstruction.

The script writes a complete traceback to PROFILE_export_error.log on failure.
"""
from __future__ import annotations

import argparse
import enum
import json
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import torch
from safetensors import safe_open

import lib.structured_convrot_full_safetensors as core

DEFAULT_FILENAMES = {
    "convrot_int8": "creaprompt_STRUCT50_CRINT8_CLIPINT4_VAEFP16.safetensors",
    "convrot_int4_mixed": "creaprompt_STRUCT50_CRINT4MIXED_CLIPINT4_VAEFP16.safetensors",
    "convrot_int4_fast": "creaprompt_STRUCT50_CRINT4FAST_CLIPINT4_VAEFP16.safetensors",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-vae-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--unet-pth", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-reconstruct-validation", action="store_true")
    parser.add_argument("--runtime-source-dir", default=None)
    return parser.parse_args()


def _json_safe(value: Any):
    """Normalize general Diffusers config values; tuples become JSON arrays."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, enum.Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    raise TypeError(f"Metadata value is not JSON serializable: {type(value)} {value!r}")


def _encode_param_value(value: Any, tensors: MutableMapping[str, torch.Tensor], key: str):
    """Encode Comfy layout params using markers understood by the core loader."""
    if isinstance(value, torch.Tensor):
        tensors[key] = core._cpu(value)
        return {"tensor": key}
    if isinstance(value, torch.dtype):
        return {"dtype": core._dtype_name(value)}
    if isinstance(value, tuple):
        return {"tuple": [_json_safe(item) for item in value]}
    return _json_safe(value)


def _architecture(unet: torch.nn.Module):
    result = core._ORIGINAL_ARCHITECTURE(unet)
    result["config"] = _json_safe(result["config"])
    return result


def _copy_source_without_unet(source: Path, output):
    """Copy owned tensors before closing the source SafeTensors mmap."""
    counts = Counter()
    with safe_open(str(source), framework="pt", device="cpu") as source_file:
        metadata = dict(source_file.metadata() or {})
        keys = list(source_file.keys())
        for index, key in enumerate(keys, 1):
            if key.startswith(core.SOURCE_UNET_PREFIX):
                counts["removed_old_unet"] += 1
                continue
            output[key] = source_file.get_tensor(key).contiguous().clone()
            if key.startswith(core.CLIP_PREFIXES):
                counts["clip"] += 1
            elif key.startswith(core.VAE_PREFIX):
                counts["vae"] += 1
            else:
                counts["other"] += 1
            if index % 500 == 0 or index == len(keys):
                print(
                    f"  copied source tensors {index}/{len(keys)} | "
                    f"clip={counts['clip']} vae={counts['vae']} other={counts['other']} "
                    f"removed_unet={counts['removed_old_unet']}",
                    flush=True,
                )
    return metadata, dict(counts)


def write_runtime_bundle(output_dir: Path, source_dir: Path):
    runtime = output_dir / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    copies = {
        source_dir / "lib" / "structured_convrot_full_safetensors.py": runtime / "structured_convrot_full_safetensors.py",
        source_dir / "lib" / "convrot_sdxl.py": runtime / "convrot_sdxl.py",
    }
    for source, target in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
    (runtime / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "requirements.txt").write_text(
        "torch\ndiffusers==0.39.0\naccelerate\nsafetensors>=0.5.2\ncomfy-kitchen>=0.2.17\n",
        encoding="utf-8",
    )
    (runtime / "README_LOAD.md").write_text(
        "SafeTensors contains the exact structured architecture metadata and all packed tensors.\n"
        "Use load_structured_convrot_unet() from structured_convrot_full_safetensors.py.\n"
        "No dense UNet or source .pth is loaded during inference.\n",
        encoding="utf-8",
    )
    return runtime


def main():
    args = parse_args()
    source = Path(args.clip_vae_source).resolve()
    pth = Path(args.unet_pth).resolve()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not pth.is_file():
        raise FileNotFoundError(pth)
    filename = DEFAULT_FILENAMES.get(args.profile, f"{args.profile}_CLIPINT4_VAEFP16.safetensors")
    output = root / filename

    print("=" * 100, flush=True)
    print(f"EXPORT {args.profile}", flush=True)
    print("Source:", source, flush=True)
    print("UNet:", pth, flush=True)
    print("Output:", output, flush=True)
    print("=" * 100, flush=True)

    result = core.build_full_checkpoint(
        pth,
        source,
        output,
        profile=args.profile,
        overwrite=args.overwrite,
    )
    print("SafeTensors write complete; starting validation", flush=True)
    validation = core.validate_full_checkpoint(
        output,
        reconstruct_unet=not args.skip_reconstruct_validation,
    )
    result["validation"] = validation

    source_dir = Path(args.runtime_source_dir).resolve() if args.runtime_source_dir else Path(__file__).resolve().parent
    runtime = write_runtime_bundle(root, source_dir)
    result["runtime_dir"] = str(runtime)
    result["dense_unet_required"] = False
    result["standard_loader_compatible"] = False
    result_path = root / f"{args.profile}_export_manifest.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"COMPLETE {args.profile}: {output.stat().st_size / 1024**3:.3f} GiB", flush=True)


core._ORIGINAL_ARCHITECTURE = core._architecture
core._encode_param_value = _encode_param_value
core._architecture = _architecture
core._copy_source_without_unet = _copy_source_without_unet


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        text = traceback.format_exc()
        try:
            values = sys.argv
            output_dir = None
            if "--output-dir" in values:
                output_dir = Path(values[values.index("--output-dir") + 1]).resolve()
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                profile = values[values.index("--profile") + 1] if "--profile" in values else "unknown"
                (output_dir / f"{profile}_export_error.log").write_text(text, encoding="utf-8")
        finally:
            print(text, file=sys.stderr, flush=True)
        raise

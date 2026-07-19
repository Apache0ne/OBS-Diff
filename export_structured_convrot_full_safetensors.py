#!/usr/bin/env python3
"""Combine structured ConvRot UNet packages with INT4 SDXL CLIPs and FP16 VAE."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch

from lib.structured_convrot_full_safetensors import build_full_checkpoint, validate_full_checkpoint

DEFAULT_FILENAMES = {
    "convrot_int8": "creaprompt_STRUCT50_CRINT8_CLIPINT4_VAEFP16.safetensors",
    "convrot_int4_mixed": "creaprompt_STRUCT50_CRINT4MIXED_CLIPINT4_VAEFP16.safetensors",
    "convrot_int4_fast": "creaprompt_STRUCT50_CRINT4FAST_CLIPINT4_VAEFP16.safetensors",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-vae-source", required=True)
    p.add_argument("--output-dir", default="/content/obs_diff_sdxl_convrot_full_safetensors")
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help="PROFILE=/absolute/path/to/unet_quantized.pth; repeat for each profile",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-reconstruct-validation", action="store_true")
    p.add_argument("--runtime-source-dir", default=None)
    return p.parse_args()


def parse_models(values):
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected PROFILE=PATH, got {value!r}")
        profile, path = value.split("=", 1)
        profile = profile.strip()
        path = Path(path.strip()).resolve()
        if not profile:
            raise ValueError(f"Empty profile in {value!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append((profile, path))
    return output


def write_runtime_bundle(root: Path, source_dir: Path | None):
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    if source_dir is None:
        source_dir = Path(__file__).resolve().parent
    source_dir = source_dir.resolve()
    files = {
        source_dir / "lib" / "structured_convrot_full_safetensors.py": runtime / "structured_convrot_full_safetensors.py",
        source_dir / "lib" / "convrot_sdxl.py": runtime / "convrot_sdxl.py",
    }
    for source, target in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
    (runtime / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "requirements.txt").write_text(
        "torch\ndiffusers==0.39.0\naccelerate\nsafetensors>=0.5.2\ncomfy-kitchen>=0.2.17\n",
        encoding="utf-8",
    )
    (runtime / "README_LOAD.md").write_text(
        "# Structured ConvRot SDXL SafeTensors runtime\n\n"
        "SafeTensors stores tensors only. `structured_convrot_full_safetensors.py` reconstructs the "
        "physically pruned UNet from explicit config/shape metadata and installs packed Comfy-Kitchen "
        "INT8/W4A4 tensors without persistent FP16 dequantization.\n\n"
        "```python\n"
        "import torch\n"
        "from runtime.structured_convrot_full_safetensors import load_structured_convrot_unet\n"
        "unet = load_structured_convrot_unet('model.safetensors', device='cuda', dtype=torch.float16)\n"
        "```\n\n"
        "The same SafeTensors also contains both original ConvRot W4A4 INT4 SDXL CLIPs and the FP16 VAE. "
        "A custom full-pipeline/ComfyUI loader is required to bind those packed CLIPs and this custom UNet; "
        "standard SDXL checkpoint loaders cannot infer the pruned architecture.\n",
        encoding="utf-8",
    )
    return runtime


def main():
    args = parse_args()
    source = Path(args.clip_vae_source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.model)
    results = []
    for profile, pth in models:
        filename = DEFAULT_FILENAMES.get(profile, f"{profile}_CLIPINT4_VAEFP16.safetensors")
        output = root / filename
        print("\n" + "=" * 100)
        print(f"EXPORT {profile}")
        print("=" * 100)
        result = build_full_checkpoint(
            pth,
            source,
            output,
            profile=profile,
            overwrite=args.overwrite,
        )
        validation = validate_full_checkpoint(
            output,
            reconstruct_unet=not args.skip_reconstruct_validation,
        )
        result["validation"] = validation
        results.append(result)
        print(json.dumps(result, indent=2))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    runtime_source = Path(args.runtime_source_dir).resolve() if args.runtime_source_dir else None
    runtime = write_runtime_bundle(root, runtime_source)
    summary = {
        "schema": "apacheone.structured_convrot_full_sdxl.v1",
        "clip_vae_source": str(source),
        "outputs": results,
        "runtime_dir": str(runtime),
        "dense_unet_required": False,
        "standard_loader_compatible": False,
    }
    summary_path = root / "structured_convrot_full_safetensors_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print("FULL SAFETENSORS EXPORT COMPLETE")
    print("=" * 100)
    for result in results:
        print(f"{result['profile']:22s} {result['bytes']/1024**3:.3f} GiB -> {result['path']}")
    print("Runtime:", runtime)
    print("Manifest:", summary_path)


if __name__ == "__main__":
    main()

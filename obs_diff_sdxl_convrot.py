#!/usr/bin/env python3
"""Quantize a structured OBS-Diff SDXL UNet with Comfy ConvRot and benchmark it.

Outputs complete standalone UNet objects for:
  * ConvRot W8A8 INT8
  * PR #14859 quality-first mixed W4A4/INT8
  * PR #14859 maximum-coverage W4A4-fast/INT8 fallback

The complete structured FP16 UNet is the comparison reference. Every variant is
loaded into the same Hyper-SDXL pipeline with identical prompts, seeds, scheduler,
resolution, steps and CFG. The report includes file/storage size, layer coverage,
latency, peak VRAM and pixel divergence. ImageReward can be added afterward by
``obs_diff_sdxl_imagereward.py``.
"""
from __future__ import annotations

import argparse
import base64
import gc
import html
import json
import math
import random
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from diffusers import DPMSolverSinglestepScheduler, StableDiffusionXLPipeline

from lib.convrot_sdxl import (
    load_comfy_quant_runtime,
    model_actual_storage_bytes,
    model_logical_parameter_count,
    quantize_unet,
    quantized_layout_counts,
    reload_verify,
    save_quantized_unet,
)

FIXED_PROMPTS = [
    "AN ADULT BEAR IS STANDING IN THE FIELD",
    "an odd looking toilet is against a wall",
    "A bathroom scene is shown with a tub and counter.",
    "a large plane is flying in the sky",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-unet", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", default="/content/obs_diff_sdxl_convrot_results")
    parser.add_argument("--profiles", default="convrot_int8,convrot_int4_mixed,convrot_int4_fast")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--coco-captions", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--quant-device", default="cuda")
    parser.add_argument("--stochastic-rounding", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-conversion", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def configure_scheduler(pipe: StableDiffusionXLPipeline) -> None:
    config = dict(pipe.scheduler.config)
    config.update(
        {
            "solver_order": 2,
            "algorithm_type": "sde-dpmsolver++",
            "solver_type": "midpoint",
            "lower_order_final": True,
            "thresholding": False,
            "use_karras_sigmas": False,
            "use_exponential_sigmas": False,
            "use_beta_sigmas": False,
            "final_sigmas_type": "zero",
            "steps_offset": 0,
        }
    )
    try:
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(config)
    except TypeError:
        for key in ("use_exponential_sigmas", "use_beta_sigmas", "final_sigmas_type"):
            config.pop(key, None)
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(config)


def load_prompts(args: argparse.Namespace) -> List[str]:
    output = list(FIXED_PROMPTS)
    if args.coco_captions and Path(args.coco_captions).is_file():
        payload = json.loads(Path(args.coco_captions).read_text(encoding="utf-8"))
        extra = [str(row.get("caption", "")).strip() for row in payload.get("annotations", [])]
        extra = [prompt for prompt in extra if prompt and prompt not in output]
        random.Random(args.seed).shuffle(extra)
        output.extend(extra)
    return output[: max(1, args.num_prompts)]


def dtype_for(args: argparse.Namespace) -> torch.dtype:
    return torch.float16 if args.dtype == "float16" else torch.bfloat16


def load_complete_unet(path: Path):
    load_comfy_quant_runtime()
    unet = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(unet, torch.nn.Module):
        raise TypeError(f"Expected complete UNet nn.Module at {path}; got {type(unet)}")
    unet.eval()
    return unet


def load_pipeline(args: argparse.Namespace, unet_path: Path) -> StableDiffusionXLPipeline:
    unet = load_complete_unet(unet_path)
    dtype = dtype_for(args)
    checkpoint = Path(args.base_checkpoint)
    kwargs = {"unet": unet, "torch_dtype": dtype, "local_files_only": args.local_files_only}
    if checkpoint.is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(checkpoint),
            use_safetensors=checkpoint.suffix.lower() == ".safetensors",
            **kwargs,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(str(checkpoint), **kwargs)
    if pipe.unet is not unet:
        raise RuntimeError("Pipeline did not retain the supplied standalone UNet override")
    configure_scheduler(pipe)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    pipe.unet.eval()
    return pipe


def file_or_zero(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def pixel_metrics(reference: str, candidate: str) -> Dict[str, float]:
    first = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float32)
    second = np.asarray(Image.open(candidate).convert("RGB"), dtype=np.float32)
    difference = second - first
    mse = float(np.mean(difference * difference))
    rmse = math.sqrt(mse)
    return {
        "psnr_db": float("inf") if mse == 0 else 20.0 * math.log10(255.0 / rmse),
        "mae_0_1": float(np.mean(np.abs(difference))) / 255.0,
        "rmse_0_1": rmse / 255.0,
        "pixels_gt_16_pct": float(np.mean(np.max(np.abs(difference), axis=2) > 16) * 100.0),
    }


def image_uri(path: str) -> str:
    buffer = BytesIO()
    Image.open(path).convert("RGB").save(buffer, "JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def benchmark_variant(
    args: argparse.Namespace,
    variant: str,
    unet_path: Path,
    prompts: Sequence[str],
    root: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 96)
    print(f"BENCHMARK {variant}: {unet_path}")
    print("=" * 96)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    pipe = load_pipeline(args, unet_path)
    load_seconds = time.perf_counter() - load_started
    current_vram = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    quant_layouts = dict(quantized_layout_counts(pipe.unet))
    output_folder = root / "images" / variant
    output_folder.mkdir(parents=True, exist_ok=True)
    rows = []
    times = []

    with torch.inference_mode():
        pipe(
            prompt=prompts[0],
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cuda").manual_seed(args.seed),
            output_type="latent",
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for index, prompt in enumerate(prompts):
        seed = args.seed + index
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        path = output_folder / f"case_{index:02d}.png"
        image.save(path)
        times.append(elapsed)
        rows.append(
            {
                "index": index,
                "prompt": prompt,
                "seed": seed,
                "path": str(path),
                "generation_seconds": elapsed,
            }
        )
        print(f"  case {index + 1}/{len(prompts)}: {elapsed:.4f}s")

    peak_vram = torch.cuda.max_memory_allocated()
    unet_parameters = model_logical_parameter_count(pipe.unet)
    actual_storage = model_actual_storage_bytes(pipe.unet)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "images": rows,
        "load_seconds": load_seconds,
        "mean_generation_seconds": float(np.mean(times)),
        "median_generation_seconds": float(np.median(times)),
        "minimum_generation_seconds": float(np.min(times)),
        "maximum_generation_seconds": float(np.max(times)),
        "loaded_vram_bytes": int(current_vram),
        "peak_vram_bytes": int(peak_vram),
        "unet_parameters": int(unet_parameters),
        "actual_model_storage_bytes": int(actual_storage),
        "quantized_layout_counts": quant_layouts,
    }


def write_html(data: Mapping[str, Any], path: Path) -> None:
    variants = list(data["variants"])
    css = (
        "body{font-family:Arial;background:#f3f4f6;margin:20px}.card{background:white;padding:15px;"
        "margin:12px 0;border:1px solid #ccc}.grid{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));"
        "gap:8px;overflow-x:auto}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #bbb;padding:6px}pre{white-space:pre-wrap;background:#111;color:#eee;padding:10px}"
    )
    output = [
        f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
        "<h1>Structured OBS-Diff SDXL + Comfy ConvRot comparison</h1>",
        "<section class='card'><p>The FP16 reference is the already-structured 50% OBS-Diff UNet. "
        "INT8 and W4A4 variants use comfy-kitchen QuantizedTensor kernels and remain complete standalone UNet objects.</p></section>",
        "<section class='card'><table><tr><th>Variant</th><th>Quantization</th><th>File GiB</th>"
        "<th>Actual model storage GiB</th><th>Quantized parameter coverage</th><th>Mean seconds</th>"
        "<th>Peak VRAM GiB</th><th>Layouts</th></tr>",
    ]
    for variant, item in data["variants"].items():
        output.append(
            f"<tr><td>{html.escape(variant)}</td><td>{html.escape(str(item.get('quantization','FP16')))}</td>"
            f"<td>{item.get('pth_bytes',0)/1024**3:.3f}</td>"
            f"<td>{item.get('actual_model_storage_bytes',0)/1024**3:.3f}</td>"
            f"<td>{100*item.get('quantized_fraction_of_unet_parameters',0):.3f}%</td>"
            f"<td>{item['mean_generation_seconds']:.4f}</td>"
            f"<td>{item.get('peak_vram_bytes',0)/1024**3:.3f}</td>"
            f"<td>{html.escape(json.dumps(item.get('quantized_layout_counts',{})))}</td></tr>"
        )
    output.append("</table></section>")
    for case in data["cases"]:
        output.append(f"<section class='card'><h2>{html.escape(case['prompt'])}</h2><div class='grid'>")
        for variant in variants:
            record = case["images"][variant]
            detail = f"{record['generation_seconds']:.3f}s"
            if variant != "dense":
                metric = case["metrics"][variant]
                detail += f" | PSNR {metric['psnr_db']:.2f} | MAE {metric['mae_0_1']:.4f}"
            output.append(
                f"<figure><b>{html.escape(variant)}</b><br><small>{detail}</small>"
                f"<img src='{image_uri(record['path'])}'></figure>"
            )
        output.append("</div></section>")
    output.append("<section class='card'><pre>" + html.escape(json.dumps(data, indent=2)) + "</pre></section>")
    output.append("</body></html>")
    path.write_text("".join(output), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    load_comfy_quant_runtime()
    structured_path = Path(args.structured_unet).resolve()
    checkpoint_path = Path(args.base_checkpoint).resolve()
    if not structured_path.is_file():
        raise FileNotFoundError(structured_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    allowed = {"convrot_int8", "convrot_int4_mixed", "convrot_int4_fast"}
    unknown = sorted(set(profiles) - allowed)
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)
    structured_manifest_path = structured_path.parent / "structured_pruning_manifest.json"
    structured_manifest = (
        json.loads(structured_manifest_path.read_text(encoding="utf-8"))
        if structured_manifest_path.is_file()
        else {}
    )
    if structured_manifest.get("unet_parameters"):
        baseline_parameters = int(structured_manifest["unet_parameters"])
    else:
        temporary = load_complete_unet(structured_path)
        baseline_parameters = model_logical_parameter_count(temporary)
        del temporary
        gc.collect()
    dense_original_parameters = int(structured_manifest.get("dense_unet_parameters", baseline_parameters))
    physical_reduction = 1.0 - baseline_parameters / dense_original_parameters

    profile_paths: Dict[str, Path] = {}
    profile_manifests: Dict[str, Dict[str, Any]] = {}
    if not args.skip_conversion:
        for profile in profiles:
            print("\n" + "=" * 96)
            print(f"CONVERT {profile}")
            print("=" * 96)
            unet = load_complete_unet(structured_path)
            before_parameters = model_logical_parameter_count(unet)
            if before_parameters != baseline_parameters:
                raise RuntimeError(
                    f"Structured UNet parameter mismatch: {before_parameters} != {baseline_parameters}"
                )
            started = time.perf_counter()
            quant_manifest = quantize_unet(
                unet,
                profile,
                quant_device=args.quant_device,
                stochastic_rounding=args.stochastic_rounding,
            )
            conversion_seconds = time.perf_counter() - started
            output_path = root / "unets" / profile / "unet_quantized.pth"
            quant_manifest.update(
                {
                    "source_structured_unet": str(structured_path),
                    "base_checkpoint": str(checkpoint_path),
                    "conversion_seconds": conversion_seconds,
                    "physical_parameter_reduction_fraction": physical_reduction,
                    "dense_original_parameters": dense_original_parameters,
                    "comfyui_formats": (
                        "int8_tensorwise+convrot and convrot_w4a4, matching merged ComfyUI PR #14859"
                    ),
                }
            )
            saved = save_quantized_unet(unet, output_path, quant_manifest)
            del unet
            gc.collect()
            torch.cuda.empty_cache()

            verified = reload_verify(output_path)
            if model_logical_parameter_count(verified) != baseline_parameters:
                raise RuntimeError(f"Reload parameter mismatch for {profile}")
            layouts = dict(quantized_layout_counts(verified))
            if not layouts:
                raise RuntimeError(f"No quantized layouts survived reload for {profile}")
            saved["reload_verified"] = True
            saved["reloaded_layout_counts"] = layouts
            (output_path.parent / "convrot_manifest.json").write_text(
                json.dumps(saved, indent=2), encoding="utf-8"
            )
            profile_paths[profile] = output_path
            profile_manifests[profile] = saved
            print(
                f"  saved {output_path} ({output_path.stat().st_size/1024**3:.3f} GiB), "
                f"actual storage {saved['actual_model_storage_bytes']/1024**3:.3f} GiB, "
                f"coverage {100*saved['quantized_fraction_of_unet_parameters']:.3f}%"
            )
            del verified
            gc.collect()
            torch.cuda.empty_cache()
    else:
        for profile in profiles:
            output_path = root / "unets" / profile / "unet_quantized.pth"
            manifest_path = output_path.parent / "convrot_manifest.json"
            if not output_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"Missing existing converted profile: {output_path}")
            profile_paths[profile] = output_path
            profile_manifests[profile] = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.skip_benchmark:
        print("Conversion complete; benchmark skipped")
        return

    variant_paths = {"dense": structured_path, **profile_paths}
    variants: Dict[str, Dict[str, Any]] = {}
    cases = [
        {"index": index, "prompt": prompt, "seed": args.seed + index, "images": {}, "metrics": {}}
        for index, prompt in enumerate(prompts)
    ]

    for variant, path in variant_paths.items():
        benchmark = benchmark_variant(args, variant, path, prompts, root)
        if variant == "dense":
            variant_info = {
                "quantization": "structured_fp16",
                "profile": "structured_fp16",
                "pth_path": str(path),
                "pth_bytes": file_or_zero(path),
                "actual_model_storage_bytes": benchmark["actual_model_storage_bytes"],
                "quantized_fraction_of_unet_parameters": 0.0,
                "parameter_reduction_fraction": physical_reduction,
                "unet_parameters": baseline_parameters,
            }
        else:
            manifest = profile_manifests[variant]
            variant_info = {
                "quantization": variant,
                "profile": variant,
                "pth_path": str(path),
                "pth_bytes": file_or_zero(path),
                "actual_model_storage_bytes": int(manifest["actual_model_storage_bytes"]),
                "quantized_fraction_of_unet_parameters": float(
                    manifest["quantized_fraction_of_unet_parameters"]
                ),
                "decision_counts": manifest.get("decision_counts", {}),
                "parameter_reduction_fraction": physical_reduction,
                "unet_parameters": baseline_parameters,
                "reload_verified": bool(manifest.get("reload_verified", False)),
            }
        variant_info.update({key: value for key, value in benchmark.items() if key != "images"})
        variants[variant] = variant_info
        for case, row in zip(cases, benchmark["images"]):
            case["images"][variant] = row

    for case in cases:
        reference = case["images"]["dense"]["path"]
        for variant in profiles:
            case["metrics"][variant] = pixel_metrics(reference, case["images"][variant]["path"])

    result = {
        "config": {
            "structured_unet": str(structured_path),
            "base_checkpoint": str(checkpoint_path),
            "profiles": profiles,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "width": args.width,
            "height": args.height,
            "seed": args.seed,
            "prompts": prompts,
            "dtype": args.dtype,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "comfyui_pr": "https://github.com/Comfy-Org/ComfyUI/pull/14859",
        },
        "variants": variants,
        "cases": cases,
    }
    json_path = root / "obs_sdxl_convrot_compare.json"
    html_path = root / "obs_sdxl_convrot_compare.html"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_html(result, html_path)

    print("\n" + "=" * 96)
    print("CONVROT COMPARISON COMPLETE")
    print("=" * 96)
    for variant, item in variants.items():
        print(
            f"{variant:22s} file={item['pth_bytes']/1024**3:.3f} GiB "
            f"storage={item['actual_model_storage_bytes']/1024**3:.3f} GiB "
            f"coverage={100*item.get('quantized_fraction_of_unet_parameters',0):.3f}% "
            f"mean={item['mean_generation_seconds']:.4f}s "
            f"peak={item['peak_vram_bytes']/1024**3:.3f} GiB"
        )
    print(f"JSON: {json_path}\nHTML: {html_path}\nUNets: {root/'unets'}")


if __name__ == "__main__":
    main()

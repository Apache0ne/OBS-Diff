#!/usr/bin/env python3
"""Build a physically smaller virtual-basis SDXL UNet from an OBS-zero teacher.

Workflow:
1. Load the 50% unstructured OBS-zero UNet as the teacher.
2. Capture paired input/output activations for every SDXL GEGLU FFN.
3. Replace each FFN with an independently distilled narrower GEGLU basis.
4. Protect sensitive FFNs by retrying at a wider basis or retaining full width.
5. Run short UNet-level replay recovery over small groups of compact FFNs.
6. Export and reload-verify a complete smaller UNet .pth.
7. Compare dense, OBS-50, and virtual-basis images at 1024 resolution.

Attention shapes remain unchanged in this first FFN-first implementation.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gc
import html
import json
import math
import random
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import (
    DPMSolverSinglestepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

from lib.virtual_basis_sdxl import (
    FFNActivationRecorder,
    UNetReplayRecorder,
    aligned_width,
    clear_gradient_checkpointing_state,
    count_parameter_bytes,
    count_parameters,
    discover_ffns,
    distill_compact_ffn,
    global_recover_student,
    module_by_path,
)


FALLBACK_PROMPTS = [
    "a studio photograph of a red fox sitting in fresh snow, detailed fur",
    "an old stone lighthouse above a stormy ocean at sunset",
    "a modern glass house in a pine forest, architectural photography",
    "a bowl of fruit on a wooden table, soft window light",
    "a vintage blue automobile parked on a city street",
    "a close portrait of an astronaut wearing a reflective helmet",
    "a small robot reading a book in a quiet library",
    "a mountain village beneath the northern lights",
    "a ceramic teapot beside a stack of books, product photograph",
    "a white horse running through a green meadow",
    "a busy night market illuminated by colorful lanterns",
    "a wooden cabin beside a frozen lake at dawn",
    "a golden retriever sleeping beside a fireplace",
    "a futuristic train moving through a desert landscape",
    "a plate of pancakes with berries and maple syrup",
    "a medieval castle on a cliff above the sea",
]

DEFAULT_TEST_PROMPTS = [
    "AN ADULT BEAR IS STANDING IN THE FIELD",
    "an odd looking toilet is against a wall",
    "A bathroom scene is shown with a tub and counter.",
    "a large plane is flying in the sky",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher-unet", required=True)
    parser.add_argument("--source-json", default=None)
    parser.add_argument(
        "--output-dir",
        default="/content/obs_diff_sdxl_virtual_basis_results",
    )
    parser.add_argument("--coco-captions", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16"], default="float16"
    )
    parser.add_argument("--calibration-prompts-512", type=int, default=8)
    parser.add_argument("--calibration-prompts-1024", type=int, default=4)
    parser.add_argument("--samples-per-ffn", type=int, default=1024)
    parser.add_argument("--tokens-per-call", type=int, default=24)
    parser.add_argument("--global-records", type=int, default=8)
    parser.add_argument("--ffn-keep", type=float, default=0.65)
    parser.add_argument("--sensitive-keep", type=float, default=0.80)
    parser.add_argument("--width-multiple", type=int, default=64)
    parser.add_argument("--minimum-basis", type=int, default=256)
    parser.add_argument("--max-local-loss", type=float, default=0.020)
    parser.add_argument("--distill-steps", type=int, default=160)
    parser.add_argument("--distill-batch", type=int, default=128)
    parser.add_argument("--distill-lr", type=float, default=3e-4)
    parser.add_argument("--distill-weight-decay", type=float, default=0.0)
    parser.add_argument("--cosine-weight", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--global-group-size", type=int, default=4)
    parser.add_argument("--global-steps-per-group", type=int, default=2)
    parser.add_argument("--global-lr", type=float, default=1e-5)
    parser.add_argument("--compare-size", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--keep-records", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    return parser.parse_args()


def configure_scheduler(pipe: StableDiffusionXLPipeline) -> None:
    config = dict(pipe.scheduler.config)
    config.update(
        dict(
            solver_order=2,
            algorithm_type="sde-dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
            thresholding=False,
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
            final_sigmas_type="zero",
            steps_offset=0,
        )
    )
    try:
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(config)
    except TypeError:
        for key in (
            "use_exponential_sigmas",
            "use_beta_sigmas",
            "final_sigmas_type",
        ):
            config.pop(key, None)
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(config)


def model_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.float16 if args.dtype == "float16" else torch.bfloat16


def load_pipe(
    args: argparse.Namespace,
    *,
    unet: UNet2DConditionModel | None = None,
) -> StableDiffusionXLPipeline:
    dtype = model_dtype(args)
    model_path = Path(args.model)
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if unet is not None:
        kwargs["unet"] = unet
    if model_path.is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            use_safetensors=model_path.suffix.lower() == ".safetensors",
            **kwargs,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_path), **kwargs
        )
    configure_scheduler(pipe)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    pipe.unet.eval()
    return pipe


def calibration_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.coco_captions and Path(args.coco_captions).is_file():
        data = json.loads(
            Path(args.coco_captions).read_text(encoding="utf-8")
        )
        prompts = [
            str(item.get("caption", "")).strip()
            for item in data.get("annotations", [])
        ]
        prompts = [prompt for prompt in prompts if prompt]
        random.Random(args.seed).shuffle(prompts)
    needed = args.calibration_prompts_512 + args.calibration_prompts_1024
    prompts = (prompts + FALLBACK_PROMPTS)[:needed]
    if len(prompts) < needed:
        raise RuntimeError("Not enough calibration prompts")
    return prompts


def pipeline_latent_pass(
    args: argparse.Namespace,
    pipe: StableDiffusionXLPipeline,
    prompts: Sequence[str],
    size: int,
    seed_offset: int,
) -> None:
    for index, prompt in enumerate(prompts):
        print(f"  capture {size} prompt {index + 1}/{len(prompts)}")
        pipe(
            prompt=prompt,
            height=size,
            width=size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cuda").manual_seed(
                args.seed + seed_offset + index
            ),
            output_type="latent",
        )


def capture_records(
    args: argparse.Namespace,
    pipe: StableDiffusionXLPipeline,
    targets,
    prompts: Sequence[str],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[Dict[str, Any]]]:
    small_count = args.calibration_prompts_512
    prompts_512 = list(prompts[:small_count])
    prompts_1024 = list(prompts[small_count:])
    recorder = FFNActivationRecorder(
        pipe.unet,
        targets,
        max_samples=args.samples_per_ffn,
        tokens_per_call=args.tokens_per_call,
        seed=args.seed,
    )
    replay = UNetReplayRecorder(
        pipe.unet,
        max_records=args.global_records,
        seed=args.seed + 99,
    )
    with recorder:
        with replay:
            pipeline_latent_pass(args, pipe, prompts_512, 512, 0)
        if prompts_1024:
            pipeline_latent_pass(args, pipe, prompts_1024, 1024, 10000)
    records = recorder.finalize()
    if not replay.records:
        raise RuntimeError("No UNet replay records were captured")
    return records, replay.records


def metrics(reference: str | Path, candidate: str | Path) -> Dict[str, float]:
    a = np.asarray(Image.open(reference).convert("RGB"), np.float32)
    b = np.asarray(Image.open(candidate).convert("RGB"), np.float32)
    delta = b - a
    mse = float(np.mean(delta * delta))
    rmse = math.sqrt(mse)
    return {
        "psnr_db": (
            float("inf") if mse == 0 else 20.0 * math.log10(255.0 / rmse)
        ),
        "mae_0_1": float(np.mean(np.abs(delta))) / 255.0,
        "rmse_0_1": rmse / 255.0,
        "pixels_gt_16_pct": float(
            np.mean(np.max(np.abs(delta), axis=2) > 16) * 100.0
        ),
    }


def image_uri(path: str | Path) -> str:
    buffer = BytesIO()
    Image.open(path).convert("RGB").save(
        buffer, "JPEG", quality=90, optimize=True
    )
    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def build_html(data: Mapping[str, Any], path: Path) -> None:
    variants = list(data["variants"].keys())
    css = (
        "body{font-family:Arial;background:#f3f4f6;margin:20px}"
        ".card{background:white;padding:15px;margin:12px 0;border:1px solid #ccc}"
        ".grid{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:10px}"
        "img{width:100%}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #bbb;padding:6px}"
        "pre{white-space:pre-wrap;background:#111;color:#eee;padding:10px}"
    )
    output = [
        f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head>"
        "<body><h1>OBS-50 teacher to virtual-basis SDXL comparison</h1>"
    ]
    output.append(
        "<section class='card'><p>The virtual-basis student uses physically narrower "
        "dense GEGLU FFNs. Attention dimensions remain unchanged. Each compact FFN "
        "was locally distilled from the 50% OBS-zero teacher and then replay-recovered "
        "against full UNet teacher outputs.</p></section>"
    )
    output.append(
        "<section class='card'><table><tr><th>Variant</th><th>UNet parameters</th>"
        "<th>Physical reduction</th><th>Parameter GiB</th><th>.pth GiB</th>"
        "<th>Mean seconds</th><th>Peak allocated GiB</th></tr>"
    )
    for variant in variants:
        row = data["variants"][variant]
        output.append(
            f"<tr><td>{html.escape(variant)}</td>"
            f"<td>{row.get('unet_parameters', 0) / 1e9:.4f}B</td>"
            f"<td>{100 * row.get('parameter_reduction_fraction', 0):.3f}%</td>"
            f"<td>{row.get('parameter_bytes', 0) / 1024**3:.3f}</td>"
            f"<td>{row.get('pth_bytes', 0) / 1024**3:.3f}</td>"
            f"<td>{row.get('mean_generation_seconds', 0):.4f}</td>"
            f"<td>{row.get('peak_allocated_bytes', 0) / 1024**3:.3f}</td></tr>"
        )
    output.append("</table></section>")
    for case in data["cases"]:
        output.append(
            f"<section class='card'><h2>{html.escape(case['prompt'])}</h2>"
            "<div class='grid'>"
        )
        for variant in variants:
            image = case["images"][variant]
            detail = f"{image['generation_seconds']:.3f}s"
            if variant != "dense":
                value = case["metrics"][variant]
                detail += (
                    f" | PSNR {value['psnr_db']:.2f}"
                    f" | MAE {value['mae_0_1']:.4f}"
                )
            output.append(
                f"<figure><b>{html.escape(variant)}</b><br><small>{detail}</small>"
                f"<img src='{image_uri(image['path'])}'></figure>"
            )
        output.append("</div></section>")
    output.append(
        "<section class='card'><pre>"
        + html.escape(json.dumps(data, indent=2))
        + "</pre></section>"
    )
    output.append("</body></html>")
    path.write_text("".join(output), encoding="utf-8")


def benchmark_latent_peak(
    args: argparse.Namespace,
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    seed: int,
) -> int:
    pipe(
        prompt=prompt,
        height=args.compare_size,
        width=args.compare_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cuda").manual_seed(seed),
        output_type="latent",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pipe(
        prompt=prompt,
        height=args.compare_size,
        width=args.compare_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cuda").manual_seed(seed),
        output_type="latent",
    )
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def generate_student(
    args: argparse.Namespace,
    student: UNet2DConditionModel,
    prompts: Sequence[str],
    output_root: Path,
) -> Tuple[List[Dict[str, Any]], float, int, int]:
    pipe = load_pipe(args, unet=student)
    latent_peak = benchmark_latent_peak(args, pipe, prompts[0], args.seed)
    folder = output_root / "images" / "virtual_basis"
    folder.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    times: List[float] = []
    peaks: List[int] = []
    for index, prompt in enumerate(prompts):
        seed = args.seed + index
        if index == 0:
            pipe(
                prompt=prompt,
                height=args.compare_size,
                width=args.compare_size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator("cuda").manual_seed(seed),
                output_type="latent",
            )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        image = pipe(
            prompt=prompt,
            height=args.compare_size,
            width=args.compare_size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated())
        image_path = folder / f"case_{index:02d}.png"
        image.save(image_path)
        rows.append(
            {
                "index": index,
                "prompt": prompt,
                "seed": seed,
                "path": str(image_path),
                "generation_seconds": elapsed,
                "peak_allocated_bytes": peak,
            }
        )
        times.append(elapsed)
        peaks.append(peak)
        print(
            f"  virtual_basis {index}: {elapsed:.3f}s "
            f"peak={peak / 1024**3:.3f} GiB"
        )
    pipe.unet.to("cpu")
    pipe.unet = None
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return rows, float(np.mean(times)), int(max(peaks)), latent_peak


def source_report(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.source_json:
        raise RuntimeError("--source-json is required for dense/OBS comparison")
    source = Path(args.source_json)
    if not source.is_file():
        raise FileNotFoundError(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    if (
        "dense" not in data.get("variants", {})
        or "sparsity_50" not in data.get("variants", {})
    ):
        raise RuntimeError(
            "Source report must contain dense and sparsity_50 variants"
        )
    return data


def save_progress(
    student: UNet2DConditionModel,
    manifest: Mapping[str, Any],
    root: Path,
) -> None:
    temporary = root / "student_progress.tmp.pth"
    final = root / "student_progress.pth"
    torch.save(student, temporary)
    temporary.replace(final)
    (root / "student_progress.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not (0.0 < args.ffn_keep <= args.sensitive_keep <= 1.0):
        raise ValueError("Require 0 < ffn_keep <= sensitive_keep <= 1")
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.output_dir).resolve()
    if root.exists() and not args.resume:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "ffn_teacher_records.pt"
    replay_path = root / "unet_replay_records.pt"
    partial_path = root / "student_progress.pth"
    progress_json = root / "student_progress.json"
    dtype = model_dtype(args)

    if args.resume and partial_path.is_file() and progress_json.is_file():
        print("Loading partial virtual-basis student")
        student = torch.load(
            partial_path, map_location="cpu", weights_only=False
        )
        progress = json.loads(progress_json.read_text(encoding="utf-8"))
        records = torch.load(
            records_path, map_location="cpu", weights_only=False
        )
        replay_records = torch.load(
            replay_path, map_location="cpu", weights_only=False
        )
    else:
        teacher_dir = Path(args.teacher_unet)
        if not teacher_dir.is_dir():
            raise FileNotFoundError(teacher_dir)
        teacher_unet = UNet2DConditionModel.from_pretrained(
            str(teacher_dir),
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )
        pipe = load_pipe(args, unet=teacher_unet)
        targets = discover_ffns(pipe.unet)
        prompts = calibration_prompts(args)
        print(f"Discovered {len(targets)} FFNs")
        records, replay_records = capture_records(
            args, pipe, targets, prompts
        )
        teacher_peak = benchmark_latent_peak(
            args, pipe, DEFAULT_TEST_PROMPTS[0], args.seed
        )
        torch.save(records, records_path)
        torch.save(replay_records, replay_path)
        student = pipe.unet.to("cpu")
        pipe.unet = None
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        progress = {
            "config": vars(args),
            "completed": [],
            "layers": {},
            "original_parameters": count_parameters(student),
            "original_parameter_bytes": count_parameter_bytes(student),
            "teacher_peak_allocated_bytes": teacher_peak,
        }
        save_progress(student, progress, root)

    targets = discover_ffns(student)
    completed = set(progress.get("completed", []))
    compressed_paths: List[str] = [
        path
        for path, info in progress.get("layers", {}).items()
        if info.get("compressed", False)
    ]

    for index, (path, target) in enumerate(targets.items(), 1):
        if path in completed:
            print(f"FFN {index}/{len(targets)} already complete: {path}")
            continue
        block = module_by_path(student, path)
        old_ff = block.ff
        record = records[path]
        target_width = aligned_width(
            target.inner_dim,
            args.ffn_keep,
            multiple=args.width_multiple,
            minimum=args.minimum_basis,
        )
        sensitive_width = aligned_width(
            target.inner_dim,
            args.sensitive_keep,
            multiple=args.width_multiple,
            minimum=args.minimum_basis,
        )
        print("\n" + "=" * 96)
        print(
            f"FFN {index}/{len(targets)} {path} "
            f"inner={target.inner_dim} target={target_width} "
            f"fallback={sensitive_width}"
        )
        print("=" * 96)

        compact, stats = distill_compact_ffn(
            old_ff,
            record,
            basis_dim=target_width,
            device="cuda",
            model_dtype=dtype,
            steps=args.distill_steps,
            batch_size=args.distill_batch,
            learning_rate=args.distill_lr,
            weight_decay=args.distill_weight_decay,
            cosine_weight=args.cosine_weight,
            validation_fraction=args.validation_fraction,
            seed=args.seed + index,
            log_every=args.log_every,
        )
        attempts = [stats]
        if (
            stats["best_validation_loss"] > args.max_local_loss
            and sensitive_width > target_width
        ):
            print(
                f"  sensitive layer retry: "
                f"val={stats['best_validation_loss']:.6f} "
                f"> {args.max_local_loss:.6f}"
            )
            del compact
            gc.collect()
            compact, retry = distill_compact_ffn(
                old_ff,
                record,
                basis_dim=sensitive_width,
                device="cuda",
                model_dtype=dtype,
                steps=args.distill_steps,
                batch_size=args.distill_batch,
                learning_rate=args.distill_lr,
                weight_decay=args.distill_weight_decay,
                cosine_weight=args.cosine_weight,
                validation_fraction=args.validation_fraction,
                seed=args.seed + 10000 + index,
                log_every=args.log_every,
            )
            attempts.append(retry)
            stats = retry

        compressed = stats["best_validation_loss"] <= args.max_local_loss
        if compressed:
            block.ff = compact
            compressed_paths.append(path)
            print(
                f"  accepted basis {stats['basis_dim']}/{target.inner_dim} "
                f"val={stats['best_validation_loss']:.6f}"
            )
        else:
            del compact
            print(
                f"  protected full-width FFN; "
                f"best val={stats['best_validation_loss']:.6f} "
                f"> {args.max_local_loss:.6f}"
            )
        completed.add(path)
        progress["completed"] = sorted(completed)
        progress.setdefault("layers", {})[path] = {
            **target.to_dict(),
            "compressed": bool(compressed),
            "attempts": attempts,
            "final_inner_dim": int(
                stats["basis_dim"] if compressed else target.inner_dim
            ),
        }
        if (
            index % max(args.checkpoint_every, 1) == 0
            or index == len(targets)
        ):
            progress["current_parameters"] = count_parameters(student)
            progress["current_parameter_bytes"] = count_parameter_bytes(
                student
            )
            save_progress(student, progress, root)
            print(f"  progress checkpoint: {partial_path}")

    print("\n" + "=" * 96)
    print("UNET-LEVEL REPLAY RECOVERY")
    print("=" * 96)
    recovery_history = global_recover_student(
        student,
        replay_records,
        compressed_paths,
        device="cuda",
        model_dtype=dtype,
        group_size=args.global_group_size,
        steps_per_group=args.global_steps_per_group,
        learning_rate=args.global_lr,
        cosine_weight=args.cosine_weight,
    )

    cleanup = clear_gradient_checkpointing_state(student)
    final_dir = root / "unets" / "virtual_basis_ffn"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_pth = final_dir / "unet_virtual_basis.pth"
    torch.save(student, final_pth)
    reloaded = torch.load(
        final_pth, map_location="cpu", weights_only=False
    )
    if count_parameters(reloaded) != count_parameters(student):
        raise RuntimeError(
            "Standalone virtual-basis reload parameter mismatch"
        )

    original_parameters = int(progress["original_parameters"])
    student_parameters = count_parameters(student)
    manifest = {
        "base_model": str(Path(args.model).resolve()),
        "teacher_unet": str(Path(args.teacher_unet).resolve()),
        "method": "OBS-50 virtual-basis FFN distillation",
        "attention_compression": "none",
        "original_parameters": original_parameters,
        "student_parameters": student_parameters,
        "parameter_reduction_fraction": (
            1.0 - student_parameters / original_parameters
        ),
        "original_parameter_bytes": int(
            progress["original_parameter_bytes"]
        ),
        "student_parameter_bytes": count_parameter_bytes(student),
        "pth_bytes": final_pth.stat().st_size,
        "reload_verified": True,
        "gradient_checkpoint_cleanup": cleanup,
        "compressed_ffns": len(compressed_paths),
        "total_ffns": len(targets),
        "layers": progress["layers"],
        "global_recovery": recovery_history,
        "config": vars(args),
    }
    manifest_path = final_dir / "virtual_basis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    del reloaded

    source = source_report(args)
    test_prompts = [case["prompt"] for case in source["cases"]]
    dense_pipe = load_pipe(args)
    dense_peak = benchmark_latent_peak(
        args, dense_pipe, test_prompts[0], args.seed
    )
    dense_pipe.unet.to("cpu")
    dense_pipe.unet = None
    del dense_pipe
    gc.collect()
    torch.cuda.empty_cache()
    rows, mean_time, peak_memory, student_latent_peak = generate_student(
        args, student, test_prompts, root
    )
    cases = copy.deepcopy(source["cases"])
    for case, row in zip(cases, rows):
        case["images"] = {
            "dense": case["images"]["dense"],
            "sparsity_50": case["images"]["sparsity_50"],
            "virtual_basis": row,
        }
        case["metrics"] = {
            "sparsity_50": case["metrics"]["sparsity_50"],
            "virtual_basis": metrics(
                case["images"]["dense"]["path"], row["path"]
            ),
        }

    dense_variant = copy.deepcopy(source["variants"]["dense"])
    teacher_variant = copy.deepcopy(source["variants"]["sparsity_50"])
    dense_variant.update(
        {
            "unet_parameters": original_parameters,
            "parameter_bytes": int(progress["original_parameter_bytes"]),
            "parameter_reduction_fraction": 0.0,
            "pth_bytes": 0,
            "peak_allocated_bytes": dense_peak,
        }
    )
    teacher_variant.update(
        {
            "unet_parameters": original_parameters,
            "parameter_bytes": int(progress["original_parameter_bytes"]),
            "parameter_reduction_fraction": 0.0,
            "pth_bytes": 0,
            "peak_allocated_bytes": int(
                progress.get("teacher_peak_allocated_bytes", 0)
            ),
        }
    )
    virtual_variant = {
        "unet_parameters": student_parameters,
        "parameter_bytes": count_parameter_bytes(student),
        "parameter_reduction_fraction": manifest[
            "parameter_reduction_fraction"
        ],
        "pth_bytes": final_pth.stat().st_size,
        "mean_generation_seconds": mean_time,
        "peak_allocated_bytes": student_latent_peak,
        "peak_full_image_allocated_bytes": peak_memory,
        "unet_path": str(final_pth),
        "manifest": str(manifest_path),
        "compressed_ffns": len(compressed_paths),
        "total_ffns": len(targets),
    }
    report = {
        "config": vars(args),
        "method": "OBS-50 teacher to adaptive virtual-basis FFN student",
        "variants": {
            "dense": dense_variant,
            "sparsity_50": teacher_variant,
            "virtual_basis": virtual_variant,
        },
        "cases": cases,
        "manifest": manifest,
    }
    raw_json = root / "obs_sdxl_virtual_basis_compare.json"
    raw_html = root / "obs_sdxl_virtual_basis_compare.html"
    raw_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    build_html(report, raw_html)

    if not args.keep_records:
        for cached_path in (records_path, replay_path):
            if cached_path.exists():
                cached_path.unlink()
    if partial_path.exists():
        partial_path.unlink()

    print("\n" + "=" * 96)
    print("VIRTUAL-BASIS SDXL COMPLETE")
    print("=" * 96)
    print(f"Original parameters: {original_parameters:,}")
    print(f"Student parameters:  {student_parameters:,}")
    print(
        f"Physical reduction:  "
        f"{100 * manifest['parameter_reduction_fraction']:.3f}%"
    )
    print(f"Compressed FFNs:     {len(compressed_paths)}/{len(targets)}")
    print(f"Standalone UNet:     {final_pth}")
    print(f"Manifest:            {manifest_path}")
    print(f"Raw JSON:            {raw_json}")
    print(f"Raw HTML:            {raw_html}")


if __name__ == "__main__":
    main()

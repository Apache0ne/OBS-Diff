#!/usr/bin/env python3
"""Fresh-Colab orchestrator for OBS-50 plus virtual-basis SDXL compression."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-output", default="/content/obs_diff_sdxl_virtual_source")
    parser.add_argument("--output-dir", default="/content/obs_diff_sdxl_virtual_basis_results")
    parser.add_argument("--coco-captions", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--obs-calibration-prompts", type=int, default=16)
    parser.add_argument("--obs-calibration-size", type=int, default=512)
    parser.add_argument("--package-hessian-gib", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--column-block", type=int, default=128)
    parser.add_argument("--ffn-keep", type=float, default=0.65)
    parser.add_argument("--sensitive-keep", type=float, default=0.80)
    parser.add_argument("--max-local-loss", type=float, default=0.020)
    parser.add_argument("--distill-steps", type=int, default=160)
    parser.add_argument("--distill-batch", type=int, default=128)
    parser.add_argument("--distill-lr", type=float, default=3e-4)
    parser.add_argument("--calibration-prompts-512", type=int, default=8)
    parser.add_argument("--calibration-prompts-1024", type=int, default=4)
    parser.add_argument("--samples-per-ffn", type=int, default=1024)
    parser.add_argument("--tokens-per-call", type=int, default=24)
    parser.add_argument("--global-records", type=int, default=8)
    parser.add_argument("--global-group-size", type=int, default=4)
    parser.add_argument("--global-steps-per-group", type=int, default=2)
    parser.add_argument("--global-lr", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    return parser.parse_args()


def run(command, env):
    print("\n" + "=" * 96)
    print("$", " ".join(map(str, command)))
    print("=" * 96 + "\n")
    subprocess.run(list(map(str, command)), check=True, env=env)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    source_output = Path(args.source_output).resolve()
    virtual_output = Path(args.output_dir).resolve()
    teacher_dir = source_output / "unets" / "sparsity_50"
    source_json = source_output / "obs_sdxl_compare.json"
    env = os.environ.copy()
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "DIFFUSERS_OFFLINE",
    ):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    if not args.resume or not teacher_dir.is_dir() or not source_json.is_file():
        zero_command = [
            sys.executable,
            "-u",
            root / "obs_diff_sdxl_colab.py",
            "--model",
            args.model,
            "--output-dir",
            source_output,
            "--ratios",
            "0.50",
            "--steps",
            args.steps,
            "--guidance-scale",
            args.guidance_scale,
            "--seed",
            args.seed,
            "--dtype",
            args.dtype,
            "--calibration-prompts",
            args.obs_calibration_prompts,
            "--calibration-size",
            args.obs_calibration_size,
            "--compare-size",
            1024,
            "--package-hessian-gib",
            args.package_hessian_gib,
            "--max-tokens",
            args.max_tokens,
            "--percdamp",
            0.01,
            "--column-block",
            args.column_block,
            "--save-unets",
        ]
        if args.coco_captions and Path(args.coco_captions).is_file():
            zero_command.extend(["--coco-captions", args.coco_captions])
        if args.local_files_only:
            zero_command.append("--local-files-only")
        if args.allow_tf32:
            zero_command.append("--allow-tf32")
        run(zero_command, env)
    else:
        print("Reusing existing OBS-50 source output")

    virtual_command = [
        sys.executable,
        "-u",
        root / "obs_diff_sdxl_virtual_basis.py",
        "--model",
        args.model,
        "--teacher-unet",
        teacher_dir,
        "--source-json",
        source_json,
        "--output-dir",
        virtual_output,
        "--steps",
        args.steps,
        "--guidance-scale",
        args.guidance_scale,
        "--seed",
        args.seed,
        "--dtype",
        args.dtype,
        "--ffn-keep",
        args.ffn_keep,
        "--sensitive-keep",
        args.sensitive_keep,
        "--max-local-loss",
        args.max_local_loss,
        "--distill-steps",
        args.distill_steps,
        "--distill-batch",
        args.distill_batch,
        "--distill-lr",
        args.distill_lr,
        "--calibration-prompts-512",
        args.calibration_prompts_512,
        "--calibration-prompts-1024",
        args.calibration_prompts_1024,
        "--samples-per-ffn",
        args.samples_per_ffn,
        "--tokens-per-call",
        args.tokens_per_call,
        "--global-records",
        args.global_records,
        "--global-group-size",
        args.global_group_size,
        "--global-steps-per-group",
        args.global_steps_per_group,
        "--global-lr",
        args.global_lr,
        "--compare-size",
        1024,
    ]
    if args.coco_captions and Path(args.coco_captions).is_file():
        virtual_command.extend(["--coco-captions", args.coco_captions])
    if args.resume:
        virtual_command.append("--resume")
    if args.local_files_only:
        virtual_command.append("--local-files-only")
    if args.allow_tf32:
        virtual_command.append("--allow-tf32")
    run(virtual_command, env)


if __name__ == "__main__":
    main()

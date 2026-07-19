#!/usr/bin/env python3
"""Resume hybrid SDXL physical export from completed calibration/search artifacts.

This entrypoint never deletes ``--output-dir``. It reuses:
- hessians/package_*/
- recovery_records.pt
- hybrid_XX_plan.json
- hybrid_XX_search_history.json

It restarts at physical deletion, FP32-safe teacher recovery, standalone export,
reload verification, comparison generation, and report creation.
"""
from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch

# Importing the Colab wrapper installs its expanded search ranges and corrected
# FP32-projection recovery function onto the shared core module.
import obs_diff_sdxl_hybrid_colab as colab_entry

core = colab_entry.core


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required resume artifact missing: {path}")
    return path


def main():
    a = core.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    torch.backends.cuda.matmul.allow_tf32 = a.allow_tf32
    torch.backends.cudnn.allow_tf32 = a.allow_tf32
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    root = Path(a.output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Existing hybrid output directory missing: {root}\n"
            "Run this resume entrypoint in the same Colab runtime as the failed run."
        )

    hessian_root = root / "hessians"
    if not hessian_root.is_dir():
        raise FileNotFoundError(f"Hessian cache missing: {hessian_root}")

    reductions = core.reduction_values(a.target_reductions)
    calibration_prompts, test_prompts = core.prompts(a)
    recovery_records = require_file(root / "recovery_records.pt")
    require_file(root / "sensitivity_curves.json")

    print("Loading dense model only to reconstruct target/package metadata...")
    dense_pipe = core.load_pipe(a)
    dense_params = core.count_params(dense_pipe.unet)
    dense_bytes = core.count_bytes(dense_pipe.unet)
    targets = core.discover(dense_pipe.unet)
    packages = core.package_targets(
        targets,
        int(a.package_hessian_gib * 1024**3),
    )

    # Confirm that the exact package layout needed by the current arguments is
    # present before performing any physical modification.
    for package_index, ids in enumerate(packages):
        package_dir = hessian_root / f"package_{package_index:03d}"
        if not package_dir.is_dir():
            raise FileNotFoundError(
                f"Resume package directory missing: {package_dir}. "
                "Use the same --package-hessian-gib as the original run."
            )
        for target_id in ids:
            require_file(package_dir / f"{core.safe_name(target_id)}.pt")

    plans = {}
    histories = {}
    for reduction in reductions:
        name = f"hybrid_{core.ratio_label(reduction)}"
        plan_path = require_file(root / f"{name}_plan.json")
        history_path = require_file(root / f"{name}_search_history.json")
        plans[name] = json.loads(plan_path.read_text(encoding="utf-8"))
        histories[name] = json.loads(history_path.read_text(encoding="utf-8"))

    # Recreate the dense comparison rows/timings. This is only four normal
    # generations; calibration and evolutionary search are not repeated.
    print("\nRegenerating dense comparison reference...")
    dense_rows, dense_time = core.generate(
        a,
        dense_pipe,
        test_prompts,
        "dense",
        root,
    )
    del dense_pipe
    gc.collect()
    torch.cuda.empty_cache()

    cases = [
        {
            "index": row["index"],
            "prompt": row["prompt"],
            "seed": row["seed"],
            "images": {"dense": row},
            "metrics": {},
        }
        for row in dense_rows
    ]
    variants = {
        "dense": {
            "unet_parameters": dense_params,
            "unet_parameter_bytes": dense_bytes,
            "parameter_reduction_fraction": 0.0,
            "pth_bytes": 0,
            "mean_generation_seconds": dense_time,
            "recovery_losses": [],
        }
    }

    for name, plan in plans.items():
        print("\n" + "=" * 92)
        print(f"RESUMED PHYSICAL EXPORT {name}")
        print("=" * 92)

        # Clear only this variant's partial products. Calibration/search caches
        # and all other plans remain untouched.
        variant_dir = root / "unets" / name
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        image_dir = root / "images" / name
        if image_dir.exists():
            shutil.rmtree(image_dir)

        pipe = core.load_pipe(a)
        model_targets = core.discover(pipe.unet)
        physical_manifest = []

        for package_index, ids in enumerate(packages):
            package_dir = hessian_root / f"package_{package_index:03d}"
            for target_id in ids:
                selected = plan[target_id]
                if selected["ratio"] <= 0:
                    continue

                hessian_path = package_dir / f"{core.safe_name(target_id)}.pt"
                H = torch.load(
                    hessian_path,
                    map_location="cuda",
                    weights_only=True,
                )
                target = model_targets[target_id]
                item = core.prune_target(
                    a,
                    pipe.unet,
                    target,
                    H,
                    selected["ratio"],
                )
                item.update(
                    {
                        "target_id": target_id,
                        "block_path": target.block_path,
                        "searched": selected,
                    }
                )
                physical_manifest.append(item)
                core.validate(pipe.unet, target)
                del H
                torch.cuda.empty_cache()

        print(f"Starting FP32-safe teacher recovery for {name}...")
        recovery_losses = core.recover_student(
            a,
            pipe.unet,
            recovery_records,
            plan,
        )

        params = core.count_params(pipe.unet)
        parameter_bytes = core.count_bytes(pipe.unet)
        reduction = 1.0 - params / dense_params

        rows, mean_time = core.generate(
            a,
            pipe,
            test_prompts,
            name,
            root,
        )

        variant_dir.mkdir(parents=True, exist_ok=True)
        pth = variant_dir / "unet_pruned.pth"
        pipe.unet.to("cpu")
        torch.save(pipe.unet, pth)
        pth_bytes = pth.stat().st_size

        old_unet = pipe.unet
        pipe.unet = None
        del old_unet
        gc.collect()
        torch.cuda.empty_cache()

        reloaded = torch.load(
            pth,
            map_location="cpu",
            weights_only=False,
        )
        if core.count_params(reloaded) != params:
            raise RuntimeError(f"Standalone reload mismatch for {name}")

        manifest_payload = {
            "base_model": str(Path(a.model).resolve()),
            "method": "Static Diff-ES-style search + OBS-Diff physical pruning",
            "unet_parameters": params,
            "dense_unet_parameters": dense_params,
            "parameter_reduction_fraction": reduction,
            "unet_parameter_bytes": parameter_bytes,
            "pth_bytes": pth_bytes,
            "reload_verified": True,
            "search_plan": plan,
            "physical_targets": physical_manifest,
            "recovery_losses": recovery_losses,
            "load_example": (
                "unet = torch.load('unet_pruned.pth', "
                "map_location='cpu', weights_only=False)"
            ),
        }
        (variant_dir / "hybrid_pruning_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2),
            encoding="utf-8",
        )

        variants[name] = {
            **manifest_payload,
            "pth_path": str(pth),
            "mean_generation_seconds": mean_time,
        }

        for case, row in zip(cases, rows):
            case["images"][name] = row
            case["metrics"][name] = core.image_metrics(
                Path(case["images"]["dense"]["path"]),
                Path(row["path"]),
            )

        print(
            f"{name}: params={params:,} reduction={reduction:.4%} "
            f"pth={pth_bytes / 1024**3:.3f} GiB reload=PASS"
        )

        del reloaded, pipe
        gc.collect()
        torch.cuda.empty_cache()

    data = {
        "config": vars(a),
        "search_design": {
            "static": True,
            "ffn_first": True,
            "attn1_max_fraction": 0.50,
            "attn2_max_fraction": 0.30,
            "protect_fraction": a.protect_fraction,
            "calibration_resolution": a.calibration_size,
            "recovery": "FP32 output-projection SGD teacher matching",
            "resumed_from_cached_calibration_and_search": True,
        },
        "variants": variants,
        "cases": cases,
        "search_histories": histories,
    }

    json_path = root / "obs_sdxl_hybrid_compare.json"
    html_path = root / "obs_sdxl_hybrid_compare.html"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    core.write_report(data, html_path)

    print("\n" + "=" * 92)
    print("HYBRID SDXL RESUME COMPLETE")
    print("=" * 92)
    for name, item in variants.items():
        print(
            f"{name:14s} params={item['unet_parameters']:,} "
            f"reduction={item['parameter_reduction_fraction']:.3%} "
            f"pth={item.get('pth_bytes', 0) / 1024**3:.3f} GiB "
            f"mean={item['mean_generation_seconds']:.4f}s"
        )
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    print(f"UNets: {root / 'unets'}")


if __name__ == "__main__":
    main()

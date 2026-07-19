#!/usr/bin/env python3
"""Run physically structured OBS-Diff pruning for SDXL at multiple ratios."""
from __future__ import annotations

import base64, gc, html, json, math, random, shutil, time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lib.structured_sdxl_core import (
    calibrate,
    count_bytes,
    count_params,
    discover,
    load_pipe,
    package_targets,
    parse_args,
    prompts,
    prune_target,
    ratio_label,
    ratio_values,
    timestep_weights,
    validate,
)


def generate(a, pipe, test_prompts, variant, root):
    folder = root / "images" / variant; folder.mkdir(parents=True, exist_ok=True)
    rows, times = [], []
    for i, prompt in enumerate(test_prompts):
        seed = a.seed + i
        if i == 0:
            pipe(prompt=prompt, height=a.compare_size, width=a.compare_size,
                 num_inference_steps=a.steps, guidance_scale=a.guidance_scale,
                 generator=torch.Generator("cuda").manual_seed(seed), output_type="latent")
        torch.cuda.synchronize(); started = time.perf_counter()
        image = pipe(prompt=prompt, height=a.compare_size, width=a.compare_size,
                     num_inference_steps=a.steps, guidance_scale=a.guidance_scale,
                     generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        path = folder / f"case_{i:02d}.png"; image.save(path); times.append(elapsed)
        rows.append({"index": i, "prompt": prompt, "seed": seed,
                     "path": str(path), "generation_seconds": elapsed})
        print(f"  {variant} case {i}: {elapsed:.3f}s")
    return rows, float(np.mean(times))


def metrics(reference, candidate):
    a = np.asarray(Image.open(reference).convert("RGB"), np.float32)
    b = np.asarray(Image.open(candidate).convert("RGB"), np.float32)
    d = b - a; mse = float(np.mean(d * d)); rmse = math.sqrt(mse)
    return {"psnr_db": float("inf") if mse == 0 else 20 * math.log10(255 / rmse),
            "mae_0_1": float(np.mean(np.abs(d))) / 255, "rmse_0_1": rmse / 255,
            "pixels_gt_16_pct": float(np.mean(np.max(np.abs(d), axis=2) > 16) * 100)}


def image_uri(path):
    buffer = BytesIO(); Image.open(path).convert("RGB").save(buffer, "JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def report(data, path):
    variants = list(data["variants"])
    css = "body{font-family:Arial;background:#f3f4f6;margin:20px}.card{background:white;padding:15px;margin:12px 0;border:1px solid #ccc}.grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:8px;overflow-x:auto}img{width:100%}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:6px}pre{white-space:pre-wrap;background:#111;color:#eee;padding:10px}"
    out = [f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
           "<h1>Structured OBS-Diff SDXL comparison</h1>",
           "<section class='card'><p>These variants physically delete FFN neurons and attention heads. Each unet_pruned.pth is a complete smaller UNet component loaded with torch.load(..., weights_only=False).</p></section>",
           "<section class='card'><table><tr><th>Variant</th><th>UNet params</th><th>Reduction</th><th>Parameter bytes</th><th>.pth bytes</th><th>Mean seconds</th><th>Reload</th></tr>"]
    for variant, item in data["variants"].items():
        out.append(f"<tr><td>{variant}</td><td>{item['unet_parameters']:,}</td><td>{100*item['parameter_reduction_fraction']:.3f}%</td><td>{item['unet_parameter_bytes']/1024**3:.3f} GiB</td><td>{item.get('pth_bytes',0)/1024**3:.3f} GiB</td><td>{item['mean_generation_seconds']:.4f}</td><td>{item.get('reload_verified',False)}</td></tr>")
    out.append("</table></section>")
    for case in data["cases"]:
        out.append(f"<section class='card'><h2>{html.escape(case['prompt'])}</h2><div class='grid'>")
        for variant in variants:
            rec = case["images"][variant]; detail = f"{rec['generation_seconds']:.3f}s"
            if variant != "dense":
                m = case["metrics"][variant]; detail += f" | PSNR {m['psnr_db']:.2f} | MAE {m['mae_0_1']:.4f}"
            out.append(f"<figure><b>{variant}</b><br><small>{detail}</small><img src='{image_uri(rec['path'])}'></figure>")
        out.append("</div></section>")
    out.append("<section class='card'><pre>" + html.escape(json.dumps(data, indent=2)) + "</pre></section></body></html>")
    path.write_text("".join(out), encoding="utf-8")


def save_reload(a, pipe, path, expected_params, prompt):
    path.parent.mkdir(parents=True, exist_ok=True)
    pipe.unet.to("cpu"); torch.save(pipe.unet, path); file_bytes = path.stat().st_size
    old = pipe.unet; pipe.unet = None; del old; gc.collect(); torch.cuda.empty_cache()
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if count_params(reloaded) != expected_params:
        raise RuntimeError("Standalone reload parameter mismatch")
    pipe.unet = reloaded.to("cuda", dtype=torch.float16 if a.dtype == "float16" else torch.bfloat16)
    pipe.unet.eval()
    with torch.inference_mode():
        pipe(prompt=prompt, height=a.calibration_size, width=a.calibration_size,
             num_inference_steps=a.steps, guidance_scale=a.guidance_scale,
             generator=torch.Generator("cuda").manual_seed(a.seed), output_type="latent")
    print("  standalone torch.load verification: PASS")
    return file_bytes


def prune_variant(a, ratio, calibration_prompts, test_prompts, root, dense_params, dense_images):
    variant = f"structured_{ratio_label(ratio)}"; pipe = load_pipe(a)
    targets = discover(pipe.unet); packages = package_targets(targets, int(a.package_hessian_gib * 1024**3))
    weights = timestep_weights(a.steps); manifest = []
    started = time.perf_counter()
    for package_index, ids in enumerate(packages):
        print("\n" + "=" * 88 + f"\n{variant} PACKAGE {package_index+1}/{len(packages)} targets={len(ids)}\n" + "=" * 88)
        hessians = calibrate(a, pipe, ids, targets, calibration_prompts, weights)
        for target_id in ids:
            target = targets[target_id]
            print(f"  prune {target_id}: kind={target.kind} ratio={ratio:.0%}")
            item = prune_target(a, pipe.unet, target, hessians[target_id], ratio)
            item.update({"target_id": target_id, "block_path": target.block_path})
            manifest.append(item); validate(pipe.unet, target)
            del hessians[target_id]; gc.collect(); torch.cuda.empty_cache()
        del hessians; gc.collect(); torch.cuda.empty_cache()
    prune_seconds = time.perf_counter() - started
    params = count_params(pipe.unet); param_bytes = count_bytes(pipe.unet)
    reduction = 1 - params / dense_params
    print(f"{variant}: {params:,} params ({params/1e9:.4f}B), physical reduction {reduction:.4%}")
    rows, mean_seconds = generate(a, pipe, test_prompts, variant, root)
    output_dir = root / "unets" / variant; pth = output_dir / "unet_pruned.pth"
    pth_bytes = save_reload(a, pipe, pth, params, test_prompts[0])
    manifest_payload = {"base_model": str(Path(a.model).resolve()), "method": "OBS-Diff structured",
                        "requested_structured_ratio": ratio, "unet_parameters": params,
                        "dense_unet_parameters": dense_params, "parameter_reduction_fraction": reduction,
                        "unet_parameter_bytes": param_bytes, "pth_bytes": pth_bytes,
                        "pruning_seconds": prune_seconds, "targets": manifest,
                        "load_example": "unet = torch.load('unet_pruned.pth', map_location='cpu', weights_only=False)"}
    (output_dir / "structured_pruning_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    del pipe; gc.collect(); torch.cuda.empty_cache()
    return rows, {**manifest_payload, "pth_path": str(pth),
                  "mean_generation_seconds": mean_seconds, "reload_verified": True}


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.matmul.allow_tf32 = a.allow_tf32
    torch.backends.cudnn.allow_tf32 = a.allow_tf32
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ratios = ratio_values(a.ratios); root = Path(a.output_dir).resolve()
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    calibration_prompts, test_prompts = prompts(a)

    dense_pipe = load_pipe(a); dense_params = count_params(dense_pipe.unet); dense_bytes = count_bytes(dense_pipe.unet)
    dense_rows, dense_time = generate(a, dense_pipe, test_prompts, "dense", root)
    cases = [{"index": row["index"], "prompt": row["prompt"], "seed": row["seed"],
              "images": {"dense": row}, "metrics": {}} for row in dense_rows]
    variants = {"dense": {"requested_structured_ratio": 0.0, "unet_parameters": dense_params,
                           "unet_parameter_bytes": dense_bytes, "parameter_reduction_fraction": 0.0,
                           "mean_generation_seconds": dense_time, "pth_path": None,
                           "pth_bytes": 0, "reload_verified": False}}
    del dense_pipe; gc.collect(); torch.cuda.empty_cache()

    for ratio in ratios:
        rows, info = prune_variant(a, ratio, calibration_prompts, test_prompts, root, dense_params, dense_rows)
        variant = f"structured_{ratio_label(ratio)}"; variants[variant] = info
        for case, row in zip(cases, rows):
            case["images"][variant] = row
            case["metrics"][variant] = metrics(case["images"]["dense"]["path"], row["path"])

    data = {"config": {"model": str(Path(a.model).resolve()), "ratios": ratios, "steps": a.steps,
                       "guidance_scale": a.guidance_scale, "seed": a.seed,
                       "calibration_prompts": calibration_prompts, "compare_prompts": test_prompts,
                       "calibration_size": a.calibration_size, "compare_size": a.compare_size,
                       "timestep_weights": timestep_weights(a.steps).tolist(), "percdamp": a.percdamp,
                       "ff_prune_chunk": a.ff_prune_chunk},
            "variants": variants, "cases": cases}
    json_path = root / "obs_sdxl_structured_compare.json"
    html_path = root / "obs_sdxl_structured_compare.html"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8"); report(data, html_path)
    print("\n" + "=" * 88 + "\nSTRUCTURED OBS-DIFF SDXL COMPLETE\n" + "=" * 88)
    for name, item in variants.items():
        print(f"{name:15s} params={item['unet_parameters']:,} reduction={100*item['parameter_reduction_fraction']:.3f}% pth={item.get('pth_bytes',0)/1024**3:.3f} GiB mean={item['mean_generation_seconds']:.4f}s")
    print(f"JSON: {json_path}\nHTML: {html_path}\nUNets: {root/'unets'}")


if __name__ == "__main__":
    main()

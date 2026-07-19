#!/usr/bin/env python3
"""Hybrid static SDXL pruning: Diff-ES-style search + OBS compensation.

This runner calibrates OBS Hessians once at deployment resolution, builds a
non-uniform per-block sensitivity/cost search space, evolves static structures
under exact physical parameter budgets, physically deletes the selected FFN
neurons and attention heads, optionally performs short teacher recovery, and
exports complete smaller ``unet_pruned.pth`` objects.

The search is static: the same smaller architecture is used at every denoising
step. FFNs receive the broadest search range, self-attention is restricted, and
cross-attention is strongly protected.
"""
from __future__ import annotations

import argparse
import base64
import gc
import html
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from lib.structured_sdxl_core import (
    Target,
    by_path,
    calibrate,
    configure_scheduler,
    count_bytes,
    count_params,
    discover,
    group_error,
    hessian_inverse,
    load_pipe,
    output_layer,
    package_targets,
    prompts,
    prune_target,
    ratio_label,
    timestep_weights,
    validate,
)


@dataclass
class Option:
    ratio: float
    remove_groups: int
    remove_columns: int
    parameter_saving: int
    normalized_obs_cost: float


@dataclass
class TargetCurve:
    target_id: str
    kind: str
    block_path: str
    attention_name: str | None
    width: int
    group_size: int
    options: List[Option]
    protected: bool = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", default="/content/obs_diff_sdxl_hybrid_results")
    p.add_argument("--target-reductions", default="0.20,0.30,0.40,0.50")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--calibration-prompts", type=int, default=12)
    p.add_argument("--calibration-size", type=int, default=1024)
    p.add_argument("--compare-size", type=int, default=1024)
    p.add_argument("--coco-captions", default=None)
    p.add_argument("--package-hessian-gib", type=float, default=0.80)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--ff-prune-chunk", type=int, default=256)
    p.add_argument("--population", type=int, default=72)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--elite", type=int, default=12)
    p.add_argument("--mutation-rate", type=float, default=0.08)
    p.add_argument("--protect-fraction", type=float, default=0.15)
    p.add_argument("--recovery-steps", type=int, default=12)
    p.add_argument("--recovery-size", type=int, default=512)
    p.add_argument("--recovery-lr", type=float, default=3e-5)
    p.add_argument("--recovery-records", type=int, default=12)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-tf32", action="store_true")
    return p.parse_args()


def reduction_values(raw: str) -> List[float]:
    values = sorted(set(float(x.strip()) for x in raw.split(",") if x.strip()))
    if not values or any(x <= 0 or x >= 0.75 for x in values):
        raise ValueError("Target reductions must satisfy 0 < value < 0.75")
    return values


def safe_name(value: str) -> str:
    return value.replace(".", "__").replace("/", "_")


def allowed_ratios(target: Target) -> List[float]:
    if target.kind == "ff":
        return [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    heads = target.width // target.group_size
    if target.attention_name == "attn2":
        max_remove = max(1, int(math.floor(heads * 0.10)))
    else:
        max_remove = max(1, int(math.floor(heads * 0.20)))
    return [k / heads for k in range(0, max_remove + 1)]


def exact_parameter_saving(unet: nn.Module, target: Target, removed_columns: int) -> int:
    if removed_columns <= 0:
        return 0
    block = by_path(unet, target.block_path)
    if target.kind == "ff":
        first: nn.Linear = block.ff.net[0].proj
        second: nn.Linear = block.ff.net[2]
        neurons = removed_columns
        saving = 2 * neurons * first.in_features
        if first.bias is not None:
            saving += 2 * neurons
        saving += neurons * second.out_features
        return int(saving)

    attn = getattr(block, target.attention_name)
    channels = removed_columns
    saving = 0
    for layer in (attn.to_q, attn.to_k, attn.to_v):
        saving += channels * layer.in_features
        if layer.bias is not None:
            saving += channels
    saving += channels * attn.to_out[0].out_features
    return int(saving)


@torch.no_grad()
def build_curve(a, unet, target: Target, H: torch.Tensor) -> TargetCurve:
    layer = output_layer(unet, target)
    W = layer.weight.detach().float()
    Hwork = H.detach().clone()
    Hinv, dead = hessian_inverse(Hwork, a.percdamp)
    if dead.any():
        W = W.clone(); W[:, dead] = 0
    scores = group_error(W, Hinv, target.group_size).detach().float()
    scores = torch.nan_to_num(scores, nan=float("inf"), posinf=float("inf"), neginf=0.0)
    ordered = torch.sort(scores).values
    finite = ordered[torch.isfinite(ordered)]
    scale = float(finite.sum().item()) if finite.numel() else 1.0
    scale = max(scale, 1e-12)
    groups = target.width // target.group_size
    options: List[Option] = []
    seen = set()
    for ratio in allowed_ratios(target):
        remove_groups = min(max(round(groups * ratio), 0), groups - 1)
        if remove_groups in seen:
            continue
        seen.add(remove_groups)
        remove_columns = remove_groups * target.group_size
        obs_cost = float(ordered[:remove_groups].sum().item() / scale) if remove_groups else 0.0
        if target.kind == "attention":
            obs_cost *= 3.0 if target.attention_name == "attn1" else 8.0
        options.append(Option(
            ratio=remove_groups / groups,
            remove_groups=remove_groups,
            remove_columns=remove_columns,
            parameter_saving=exact_parameter_saving(unet, target, remove_columns),
            normalized_obs_cost=obs_cost,
        ))
    del Hinv, Hwork
    torch.cuda.empty_cache()
    return TargetCurve(
        target_id=target.target_id,
        kind=target.kind,
        block_path=target.block_path,
        attention_name=target.attention_name,
        width=target.width,
        group_size=target.group_size,
        options=options,
    )


def protect_sensitive(curves: List[TargetCurve], fraction: float):
    fraction = max(0.0, min(0.75, float(fraction)))
    for kind in ("ff", "attention"):
        subset = [c for c in curves if c.kind == kind and len(c.options) > 1]
        if not subset:
            continue
        def sensitivity(c: TargetCurve):
            option = c.options[1]
            return option.normalized_obs_cost / max(option.parameter_saving, 1)
        subset.sort(key=sensitivity, reverse=True)
        count = int(round(len(subset) * fraction))
        for curve in subset[:count]:
            curve.protected = True
            curve.options = curve.options[:1]


def chromosome_stats(chromosome: Sequence[int], curves: Sequence[TargetCurve]):
    saving = 0
    cost = 0.0
    ff_saving = 0
    attn1_saving = 0
    attn2_saving = 0
    for gene, curve in zip(chromosome, curves):
        option = curve.options[gene]
        saving += option.parameter_saving
        cost += option.normalized_obs_cost
        if curve.kind == "ff":
            ff_saving += option.parameter_saving
        elif curve.attention_name == "attn1":
            attn1_saving += option.parameter_saving
        else:
            attn2_saving += option.parameter_saving
    return saving, cost, ff_saving, attn1_saving, attn2_saving


def fitness(chromosome, curves, target_saving, dense_params):
    saving, cost, _, _, _ = chromosome_stats(chromosome, curves)
    shortfall = max(0, target_saving - saving) / dense_params
    excess = max(0, saving - target_saving) / dense_params
    return cost + 2.0e5 * shortfall * shortfall + 2.0 * excess


def greedy_seed(curves, target_saving):
    genes = [0] * len(curves)
    saving = 0
    while saving < target_saving:
        best = None
        for i, curve in enumerate(curves):
            g = genes[i]
            if g + 1 >= len(curve.options):
                continue
            old = curve.options[g]; new = curve.options[g + 1]
            ds = new.parameter_saving - old.parameter_saving
            dc = new.normalized_obs_cost - old.normalized_obs_cost
            if ds <= 0:
                continue
            kind_bias = 1.0 if curve.kind == "ff" else (2.0 if curve.attention_name == "attn1" else 5.0)
            score = kind_bias * dc / ds
            if best is None or score < best[0]:
                best = (score, i, ds)
        if best is None:
            break
        _, i, ds = best
        genes[i] += 1; saving += ds
    return genes


def mutate(rng: random.Random, chromosome, curves, rate):
    child = list(chromosome)
    for i, curve in enumerate(curves):
        if len(curve.options) <= 1 or rng.random() >= rate:
            continue
        step = rng.choice([-1, 1, 1])
        child[i] = max(0, min(len(curve.options) - 1, child[i] + step))
    return child


def crossover(rng: random.Random, a, b):
    return [x if rng.random() < 0.5 else y for x, y in zip(a, b)]


def evolve(a, curves, dense_params, target_fraction):
    rng = random.Random(a.seed + int(target_fraction * 10000))
    target_saving = int(round(dense_params * target_fraction))
    maximum = sum(c.options[-1].parameter_saving for c in curves)
    if maximum < target_saving:
        raise RuntimeError(
            f"Search space can remove only {maximum/dense_params:.2%}, below requested {target_fraction:.2%}"
        )
    seed = greedy_seed(curves, target_saving)
    population = [seed]
    while len(population) < a.population:
        population.append(mutate(rng, seed, curves, rate=max(a.mutation_rate, 0.20)))
    history = []
    for generation in range(a.generations):
        ranked = sorted(
            ((fitness(ch, curves, target_saving, dense_params), ch) for ch in population),
            key=lambda item: item[0],
        )
        best_score, best = ranked[0]
        saving, cost, ffs, a1s, a2s = chromosome_stats(best, curves)
        history.append({
            "generation": generation,
            "fitness": best_score,
            "predicted_obs_cost": cost,
            "saving": saving,
            "reduction_fraction": saving / dense_params,
            "ff_saving": ffs,
            "attn1_saving": a1s,
            "attn2_saving": a2s,
        })
        if generation % 10 == 0 or generation + 1 == a.generations:
            print(
                f"  generation {generation:03d}: reduction={saving/dense_params:.4%} "
                f"cost={cost:.8f} fitness={best_score:.8f}"
            )
        elites = [ch for _, ch in ranked[: max(2, min(a.elite, len(ranked)))]]
        next_population = [list(ch) for ch in elites]
        while len(next_population) < a.population:
            p1, p2 = rng.choice(elites), rng.choice(elites)
            child = crossover(rng, p1, p2)
            child = mutate(rng, child, curves, a.mutation_rate)
            next_population.append(child)
        population = next_population
    ranked = sorted(
        ((fitness(ch, curves, target_saving, dense_params), ch) for ch in population),
        key=lambda item: item[0],
    )
    best = ranked[0][1]
    plan = {}
    for gene, curve in zip(best, curves):
        option = curve.options[gene]
        plan[curve.target_id] = {
            "ratio": option.ratio,
            "remove_groups": option.remove_groups,
            "remove_columns": option.remove_columns,
            "parameter_saving": option.parameter_saving,
            "predicted_obs_cost": option.normalized_obs_cost,
            "kind": curve.kind,
            "attention_name": curve.attention_name,
            "protected": curve.protected,
        }
    return plan, history


def image_metrics(reference: Path, candidate: Path):
    a = np.asarray(Image.open(reference).convert("RGB"), np.float32)
    b = np.asarray(Image.open(candidate).convert("RGB"), np.float32)
    d = b - a; mse = float(np.mean(d * d)); rmse = math.sqrt(mse)
    return {
        "psnr_db": float("inf") if mse == 0 else 20 * math.log10(255 / rmse),
        "mae_0_1": float(np.mean(np.abs(d))) / 255,
        "rmse_0_1": rmse / 255,
        "pixels_gt_16_pct": float(np.mean(np.max(np.abs(d), axis=2) > 16) * 100),
    }


def generate(a, pipe, test_prompts, variant, root):
    folder = root / "images" / variant; folder.mkdir(parents=True, exist_ok=True)
    rows, timings = [], []
    for i, prompt in enumerate(test_prompts):
        seed = a.seed + i
        if i == 0:
            pipe(
                prompt=prompt, height=a.compare_size, width=a.compare_size,
                num_inference_steps=a.steps, guidance_scale=a.guidance_scale,
                generator=torch.Generator("cuda").manual_seed(seed), output_type="latent",
            )
        torch.cuda.synchronize(); started = time.perf_counter()
        image = pipe(
            prompt=prompt, height=a.compare_size, width=a.compare_size,
            num_inference_steps=a.steps, guidance_scale=a.guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        path = folder / f"case_{i:02d}.png"; image.save(path); timings.append(elapsed)
        rows.append({"index": i, "prompt": prompt, "seed": seed, "path": str(path), "generation_seconds": elapsed})
        print(f"  {variant} case {i}: {elapsed:.3f}s")
    return rows, float(np.mean(timings))


@torch.no_grad()
def make_recovery_records(a, pipe, calibration_prompts, root):
    if a.recovery_steps <= 0 or a.recovery_records <= 0:
        return None
    path = root / "recovery_records.pt"
    pipe.scheduler.set_timesteps(a.steps, device="cuda")
    timesteps = pipe.scheduler.timesteps
    dtype = next(pipe.unet.parameters()).dtype
    records = []
    size = a.recovery_size
    for i in range(a.recovery_records):
        prompt = calibration_prompts[i % len(calibration_prompts)]
        prompt_embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=prompt, device="cuda", num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        time_ids = pipe._get_add_time_ids(
            (size, size), (0, 0), (size, size), dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to("cuda")
        t = timesteps[i % len(timesteps)].reshape(1)
        latent = torch.randn(
            (1, pipe.unet.config.in_channels, size // pipe.vae_scale_factor, size // pipe.vae_scale_factor),
            generator=torch.Generator("cuda").manual_seed(a.seed + 10000 + i),
            device="cuda", dtype=dtype,
        ) * pipe.scheduler.init_noise_sigma
        sample = pipe.scheduler.scale_model_input(latent, t)
        target = pipe.unet(
            sample, t, encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        ).sample
        records.append({
            "sample": sample.cpu(), "t": t.cpu(),
            "encoder_hidden_states": prompt_embeds.cpu(),
            "text_embeds": pooled.cpu(), "time_ids": time_ids.cpu(),
            "target": target.cpu(),
        })
        print(f"  recovery teacher record {i+1}/{a.recovery_records}")
    torch.save(records, path)
    return path


def recover_student(a, unet, record_path: Path | None, plan):
    if a.recovery_steps <= 0 or record_path is None:
        return []
    from transformers.optimization import Adafactor
    records = torch.load(record_path, map_location="cpu", weights_only=False)
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    trainable = []
    seen = set()
    for target_id, selected in plan.items():
        if selected["ratio"] <= 0:
            continue
        block_path = target_id.rsplit(".", 1)[0]
        block = by_path(unet, block_path)
        module = block.ff.net[2] if selected["kind"] == "ff" else getattr(block, selected["attention_name"]).to_out[0]
        for parameter in module.parameters():
            if id(parameter) not in seen:
                parameter.requires_grad_(True); trainable.append(parameter); seen.add(id(parameter))
    if not trainable:
        return []
    unet.enable_gradient_checkpointing(); unet.train()
    optimizer = Adafactor(
        trainable, lr=a.recovery_lr, scale_parameter=False,
        relative_step=False, warmup_init=False, weight_decay=0.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=a.dtype == "float16")
    losses = []
    for step in range(a.recovery_steps):
        record = records[step % len(records)]
        optimizer.zero_grad(set_to_none=True)
        dtype = next(unet.parameters()).dtype
        sample = record["sample"].to("cuda", dtype=dtype)
        t = record["t"].to("cuda")
        enc = record["encoder_hidden_states"].to("cuda", dtype=dtype)
        text = record["text_embeds"].to("cuda", dtype=dtype)
        time_ids = record["time_ids"].to("cuda", dtype=dtype)
        target = record["target"].to("cuda", dtype=torch.float32)
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            prediction = unet(
                sample, t, encoder_hidden_states=enc,
                added_cond_kwargs={"text_embeds": text, "time_ids": time_ids},
            ).sample
            loss = torch.nn.functional.mse_loss(prediction.float(), target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer); scaler.update()
        losses.append(float(loss.item()))
        print(f"  recovery step {step+1:03d}/{a.recovery_steps}: loss={losses[-1]:.8f}")
    unet.eval()
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    return losses


def image_uri(path):
    buffer = BytesIO(); Image.open(path).convert("RGB").save(buffer, "JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def write_report(data, path: Path):
    variants = list(data["variants"])
    css = "body{font-family:Arial;background:#f3f4f6;margin:20px}.card{background:white;padding:15px;margin:12px 0;border:1px solid #bbb}.grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:8px;overflow-x:auto}img{width:100%}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:6px}pre{white-space:pre-wrap;background:#111;color:#eee;padding:10px}"
    out = [f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body><h1>Hybrid static Diff-ES + OBS-Diff SDXL</h1>"]
    out.append("<section class='card'><p>Non-uniform static architecture search, FFN-first allocation, protected sensitive targets, restricted attention pruning, OBS compensation, physical deletion, and optional teacher recovery.</p><table><tr><th>Variant</th><th>Params</th><th>Physical reduction</th><th>.pth GiB</th><th>Mean seconds</th><th>Recovery final loss</th></tr>")
    for name, item in data["variants"].items():
        losses = item.get("recovery_losses", [])
        final_loss = losses[-1] if losses else None
        out.append(f"<tr><td>{name}</td><td>{item['unet_parameters']:,}</td><td>{100*item['parameter_reduction_fraction']:.3f}%</td><td>{item.get('pth_bytes',0)/1024**3:.3f}</td><td>{item['mean_generation_seconds']:.4f}</td><td>{final_loss}</td></tr>")
    out.append("</table></section>")
    for case in data["cases"]:
        out.append(f"<section class='card'><h2>{html.escape(case['prompt'])}</h2><div class='grid'>")
        for variant in variants:
            rec = case["images"][variant]; detail = f"{rec['generation_seconds']:.3f}s"
            if variant != "dense":
                metric = case["metrics"][variant]
                detail += f" | PSNR {metric['psnr_db']:.2f} | MAE {metric['mae_0_1']:.4f}"
            out.append(f"<figure><b>{variant}</b><br><small>{detail}</small><img src='{image_uri(rec['path'])}'></figure>")
        out.append("</div></section>")
    out.append("<section class='card'><pre>" + html.escape(json.dumps(data, indent=2)) + "</pre></section></body></html>")
    path.write_text("".join(out), encoding="utf-8")


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.matmul.allow_tf32 = a.allow_tf32
    torch.backends.cudnn.allow_tf32 = a.allow_tf32
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    reductions = reduction_values(a.target_reductions)
    root = Path(a.output_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    hessian_root = root / "hessians"; hessian_root.mkdir()
    calibration_prompts, test_prompts = prompts(a)

    print("Loading dense model for calibration and teacher records...")
    dense_pipe = load_pipe(a)
    dense_params = count_params(dense_pipe.unet)
    dense_bytes = count_bytes(dense_pipe.unet)
    targets = discover(dense_pipe.unet)
    packages = package_targets(targets, int(a.package_hessian_gib * 1024**3))
    weights = timestep_weights(a.steps)
    curves: List[TargetCurve] = []

    for package_index, ids in enumerate(packages):
        print("\n" + "=" * 92 + f"\nHYBRID CALIBRATION PACKAGE {package_index+1}/{len(packages)} targets={len(ids)}\n" + "=" * 92)
        hessians = calibrate(a, dense_pipe, ids, targets, calibration_prompts, weights)
        package_dir = hessian_root / f"package_{package_index:03d}"; package_dir.mkdir()
        for target_id in ids:
            H = hessians[target_id]
            curve = build_curve(a, dense_pipe.unet, targets[target_id], H)
            curves.append(curve)
            torch.save(H.detach().cpu(), package_dir / f"{safe_name(target_id)}.pt")
            del H
        del hessians; gc.collect(); torch.cuda.empty_cache()

    protect_sensitive(curves, a.protect_fraction)
    curve_lookup = {curve.target_id: curve for curve in curves}
    curve_json = []
    for curve in curves:
        payload = asdict(curve)
        curve_json.append(payload)
    (root / "sensitivity_curves.json").write_text(json.dumps(curve_json, indent=2), encoding="utf-8")

    dense_rows, dense_time = generate(a, dense_pipe, test_prompts, "dense", root)
    recovery_records = make_recovery_records(a, dense_pipe, calibration_prompts, root)
    del dense_pipe; gc.collect(); torch.cuda.empty_cache()

    plans = {}
    histories = {}
    for reduction in reductions:
        print("\n" + "=" * 92 + f"\nSTATIC EVOLUTIONARY SEARCH target={reduction:.0%}\n" + "=" * 92)
        plan, history = evolve(a, curves, dense_params, reduction)
        name = f"hybrid_{ratio_label(reduction)}"
        plans[name] = plan; histories[name] = history
        (root / f"{name}_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        (root / f"{name}_search_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    cases = [{"index": row["index"], "prompt": row["prompt"], "seed": row["seed"], "images": {"dense": row}, "metrics": {}} for row in dense_rows]
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
        print("\n" + "=" * 92 + f"\nPHYSICAL EXPORT {name}\n" + "=" * 92)
        pipe = load_pipe(a)
        model_targets = discover(pipe.unet)
        manifest = []
        for package_index, ids in enumerate(packages):
            package_dir = hessian_root / f"package_{package_index:03d}"
            for target_id in ids:
                selected = plan[target_id]
                if selected["ratio"] <= 0:
                    continue
                H = torch.load(package_dir / f"{safe_name(target_id)}.pt", map_location="cuda", weights_only=True)
                item = prune_target(a, pipe.unet, model_targets[target_id], H, selected["ratio"])
                item.update({"target_id": target_id, "block_path": model_targets[target_id].block_path, "searched": selected})
                manifest.append(item); validate(pipe.unet, model_targets[target_id])
                del H; torch.cuda.empty_cache()

        recovery_losses = recover_student(a, pipe.unet, recovery_records, plan)
        params = count_params(pipe.unet); parameter_bytes = count_bytes(pipe.unet)
        reduction = 1.0 - params / dense_params
        rows, mean_time = generate(a, pipe, test_prompts, name, root)
        output_dir = root / "unets" / name; output_dir.mkdir(parents=True, exist_ok=True)
        pth = output_dir / "unet_pruned.pth"
        pipe.unet.to("cpu"); torch.save(pipe.unet, pth); pth_bytes = pth.stat().st_size
        old = pipe.unet; pipe.unet = None; del old; gc.collect(); torch.cuda.empty_cache()
        reloaded = torch.load(pth, map_location="cpu", weights_only=False)
        if count_params(reloaded) != params:
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
            "physical_targets": manifest,
            "recovery_losses": recovery_losses,
            "load_example": "unet = torch.load('unet_pruned.pth', map_location='cpu', weights_only=False)",
        }
        (output_dir / "hybrid_pruning_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        variants[name] = {**manifest_payload, "pth_path": str(pth), "mean_generation_seconds": mean_time}
        for case, row in zip(cases, rows):
            case["images"][name] = row
            case["metrics"][name] = image_metrics(Path(case["images"]["dense"]["path"]), Path(row["path"]))
        del reloaded, pipe; gc.collect(); torch.cuda.empty_cache()

    data = {
        "config": vars(a),
        "search_design": {
            "static": True,
            "ffn_first": True,
            "attn1_max_fraction": 0.20,
            "attn2_max_fraction": 0.10,
            "protect_fraction": a.protect_fraction,
            "calibration_resolution": a.calibration_size,
            "recovery": "output-projection Adafactor teacher matching" if a.recovery_steps > 0 else "disabled",
        },
        "variants": variants,
        "cases": cases,
        "search_histories": histories,
    }
    json_path = root / "obs_sdxl_hybrid_compare.json"
    html_path = root / "obs_sdxl_hybrid_compare.html"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_report(data, html_path)
    print("\n" + "=" * 92 + "\nHYBRID SDXL COMPLETE\n" + "=" * 92)
    for name, item in variants.items():
        print(
            f"{name:14s} params={item['unet_parameters']:,} "
            f"reduction={item['parameter_reduction_fraction']:.3%} "
            f"pth={item.get('pth_bytes',0)/1024**3:.3f} GiB "
            f"mean={item['mean_generation_seconds']:.4f}s"
        )
    print(f"JSON: {json_path}\nHTML: {html_path}\nUNets: {root/'unets'}")


if __name__ == "__main__":
    main()

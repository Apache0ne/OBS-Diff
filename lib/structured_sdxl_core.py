#!/usr/bin/env python3
"""Physically structured OBS-Diff pruning for SDXL.

Each requested ratio is pruned independently from the original dense SDXL
checkpoint. Timestep-aware Hessians are collected package-by-package, OBS error
compensation is applied to FFN output columns and attention output-head blocks,
then matching GEGLU rows and Q/K/V rows are physically deleted. The result is a
complete smaller UNet saved with torch.save, matching the deployment convention
of the official Alrightlone/OBS-Diff-SDXL release.
"""
from __future__ import annotations

import argparse, base64, gc, html, json, math, random, shutil, time
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import DPMSolverSinglestepScheduler, StableDiffusionXLPipeline
from diffusers.models.activations import GEGLU
from diffusers.models.attention_processor import Attention

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
COMPARE_PROMPTS = [
    "AN ADULT BEAR IS STANDING IN THE FIELD",
    "an odd looking toilet is against a wall",
    "A bathroom scene is shown with a tub and counter.",
    "a large plane is flying in the sky",
]
STEP = {"value": 0}


@dataclass(frozen=True)
class Target:
    target_id: str
    block_path: str
    kind: str
    attention_name: str | None
    width: int
    group_size: int


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", default="/content/obs_diff_sdxl_structured_results")
    p.add_argument("--ratios", default="0.20,0.30,0.40,0.50")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--calibration-prompts", type=int, default=16)
    p.add_argument("--calibration-size", type=int, default=512)
    p.add_argument("--compare-size", type=int, default=1024)
    p.add_argument("--coco-captions", default=None)
    p.add_argument("--package-hessian-gib", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--ff-prune-chunk", type=int, default=512)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-tf32", action="store_true")
    return p.parse_args()


def ratio_values(raw: str) -> List[float]:
    values = sorted(set(float(x.strip()) for x in raw.split(",") if x.strip()))
    if not values or any(x <= 0 or x >= 1 for x in values):
        raise ValueError("Ratios must satisfy 0 < ratio < 1")
    return values


def ratio_label(r: float) -> str:
    return f"{int(round(100 * r)):02d}"


def configure_scheduler(pipe):
    cfg = dict(pipe.scheduler.config)
    cfg.update({
        "solver_order": 2, "algorithm_type": "sde-dpmsolver++", "solver_type": "midpoint",
        "lower_order_final": True, "thresholding": False, "use_karras_sigmas": False,
        "use_exponential_sigmas": False, "use_beta_sigmas": False,
        "final_sigmas_type": "zero", "steps_offset": 0,
    })
    try:
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(cfg)
    except TypeError:
        for key in ("use_exponential_sigmas", "use_beta_sigmas", "final_sigmas_type"):
            cfg.pop(key, None)
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(cfg)


def load_pipe(a):
    dtype = torch.float16 if a.dtype == "float16" else torch.bfloat16
    path = Path(a.model)
    kwargs = {"torch_dtype": dtype, "local_files_only": a.local_files_only}
    if path.is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(path), use_safetensors=path.suffix.lower() == ".safetensors", **kwargs
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(str(path), **kwargs)
    configure_scheduler(pipe)
    pipe.vae.enable_tiling(); pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda"); pipe.unet.eval()
    return pipe


def prompts(a):
    calibration: List[str] = []
    if a.coco_captions and Path(a.coco_captions).is_file():
        data = json.loads(Path(a.coco_captions).read_text(encoding="utf-8"))
        calibration = [str(x.get("caption", "")).strip() for x in data.get("annotations", [])]
        calibration = [x for x in calibration if x]
        random.Random(a.seed).shuffle(calibration)
    calibration = (calibration + FALLBACK_PROMPTS)[: a.calibration_prompts]
    if len(calibration) != a.calibration_prompts:
        raise RuntimeError("Not enough calibration prompts")
    return calibration, list(COMPARE_PROMPTS)


def by_path(root: nn.Module, path: str):
    value: Any = root
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def discover(unet) -> "OrderedDict[str, Target]":
    result: "OrderedDict[str, Target]" = OrderedDict()
    for path, block in unet.named_modules():
        if ".transformer_blocks." not in f".{path}" or not hasattr(block, "ff"):
            continue
        if not isinstance(block.ff.net[0], GEGLU) or not isinstance(block.ff.net[2], nn.Linear):
            raise RuntimeError(f"Unsupported SDXL feed-forward structure at {path}")
        result[path + ".ff"] = Target(
            path + ".ff", path, "ff", None, block.ff.net[2].in_features, 1
        )
        for name in ("attn1", "attn2"):
            attn = getattr(block, name, None)
            if attn is None:
                continue
            if not isinstance(attn, Attention) or attn.to_out is None:
                raise RuntimeError(f"Unsupported attention at {path}.{name}")
            if attn.to_q.out_features % attn.heads:
                raise RuntimeError(f"Invalid head shape at {path}.{name}")
            head_dim = attn.to_q.out_features // attn.heads
            if attn.to_k.out_features != attn.to_q.out_features or attn.to_v.out_features != attn.to_q.out_features:
                raise RuntimeError(f"GQA/MQA is unsupported at {path}.{name}")
            if attn.norm_q is not None or attn.norm_k is not None:
                raise RuntimeError(f"Q/K norm pruning is unsupported at {path}.{name}")
            result[path + "." + name] = Target(
                path + "." + name, path, "attention", name, attn.to_out[0].in_features, head_dim
            )
    if not result:
        raise RuntimeError("No SDXL structured targets found")
    return result


def output_layer(unet, target: Target) -> nn.Linear:
    block = by_path(unet, target.block_path)
    return block.ff.net[2] if target.kind == "ff" else getattr(block, target.attention_name).to_out[0]


def package_targets(targets: Mapping[str, Target], budget: int) -> List[List[str]]:
    packages, current, used = [], [], 0
    for target_id, target in targets.items():
        cost = target.width * target.width * 4
        if current and used + cost > budget:
            packages.append(current); current, used = [], 0
        current.append(target_id); used += cost
    if current:
        packages.append(current)
    return packages


def timestep_weights(steps: int):
    if steps <= 1:
        return np.ones(steps)
    x = np.arange(steps)
    return (0.8 + 0.4 / np.log(steps) * np.log1p(x))[::-1].copy()


def callback(pipe, step, timestep, kwargs):
    STEP["value"] = int(step) + 1
    return kwargs


class Hessian:
    def __init__(self, width: int, max_tokens: int, device):
        self.width, self.max_tokens = width, max_tokens
        self.H = torch.zeros((width, width), dtype=torch.float32, device=device)
        self.total_weight = 0.0; self.calls = 0; self.tokens = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor, weight: float):
        x = x.reshape(-1, x.shape[-1])
        if x.shape[1] != self.width:
            raise RuntimeError("Activation width mismatch")
        if self.max_tokens and x.shape[0] > self.max_tokens:
            idx = torch.linspace(0, x.shape[0] - 1, self.max_tokens, device=x.device).round().long()
            x = x.index_select(0, idx)
        x = x.float(); added = weight * x.shape[0]; total = self.total_weight + added
        if self.total_weight:
            self.H.mul_(self.total_weight / total)
        x.mul_(math.sqrt(2.0 * weight / total)); self.H.addmm_(x.t(), x)
        self.total_weight = total; self.calls += 1; self.tokens += x.shape[0]


@torch.no_grad()
def calibrate(a, pipe, ids, targets, calibration_prompts, weights):
    stats: Dict[str, Hessian] = {}; hooks = []
    for target_id in ids:
        layer = output_layer(pipe.unet, targets[target_id])
        stat = Hessian(layer.in_features, a.max_tokens, layer.weight.device)
        stats[target_id] = stat
        def hook(module, inputs, output, stat=stat):
            step = min(STEP["value"], len(weights) - 1)
            stat.add(inputs[0].detach(), float(weights[step]))
        hooks.append(layer.register_forward_hook(hook))
    try:
        for i, prompt in enumerate(calibration_prompts):
            STEP["value"] = 0
            pipe(
                prompt=prompt, height=a.calibration_size, width=a.calibration_size,
                num_inference_steps=a.steps, guidance_scale=a.guidance_scale,
                generator=torch.Generator("cuda").manual_seed(a.seed + i), output_type="latent",
                callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
            )
    finally:
        for hook in hooks:
            hook.remove()
    output = {}
    for target_id, stat in stats.items():
        if stat.total_weight <= 0:
            raise RuntimeError(f"No activations reached {target_id}")
        print(f"  {target_id}: H={stat.width} calls={stat.calls} sampled_tokens={stat.tokens}")
        output[target_id] = stat.H
    return output


def hessian_inverse(H: torch.Tensor, percdamp: float):
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    diagonal = torch.arange(H.shape[0], device=H.device)
    base = max(float(torch.mean(torch.diag(H))) * percdamp, 1e-8)
    error = None
    for multiplier in (1, 10, 100, 1000):
        trial = H.clone(); trial[diagonal, diagonal] += base * multiplier
        try:
            return torch.cholesky_inverse(torch.linalg.cholesky(trial)), dead
        except Exception as exc:
            error = exc; del trial; torch.cuda.empty_cache()
    raise RuntimeError("Hessian Cholesky failed") from error


def group_error(weight, Hinv, group_size):
    if group_size == 1:
        denominator = torch.diag(Hinv).clamp_min(1e-20)
        return torch.sum(weight.square() / denominator.unsqueeze(0), dim=0)
    blocks = torch.stack([
        Hinv[i:i + group_size, i:i + group_size]
        for i in range(0, Hinv.shape[0], group_size)
    ])
    denominator = torch.diagonal(torch.linalg.cholesky(blocks), dim1=-2, dim2=-1)
    denominator = denominator.reshape(-1).square().clamp_min(1e-20)
    columns = torch.sum(weight.square() / denominator.unsqueeze(0), dim=0)
    return columns.view(-1, group_size).sum(1)


@torch.no_grad()
def obs_remove_columns(weight: torch.Tensor, H: torch.Tensor, ratio: float, group_size: int, percdamp: float, ff_chunk: int):
    W = weight.detach().float().clone(); width = W.shape[1]
    if width % group_size:
        raise RuntimeError("Weight width is not group aligned")
    groups = width // group_size
    target_groups = min(max(round(groups * ratio), 0), groups - 1)
    removed_columns = torch.zeros(width, dtype=torch.bool, device=W.device)
    removed_groups = torch.zeros(groups, dtype=torch.bool, device=W.device)
    current = 0
    Hinv, dead = hessian_inverse(H, percdamp)
    W[:, dead] = 0
    del Hinv

    while current < target_groups:
        count_groups = 1 if group_size > 1 else min(ff_chunk, target_groups - current)
        Hinv, _ = hessian_inverse(H, percdamp=percdamp)
        errors = group_error(W, Hinv, group_size); errors[removed_groups] = torch.inf
        chosen_groups = torch.topk(errors, k=count_groups, largest=False).indices
        chosen_columns = torch.cat([
            torch.arange(g * group_size, (g + 1) * group_size, device=W.device)
            for g in chosen_groups.tolist()
        ])
        live_mask = ~removed_columns; live_mask[chosen_columns] = False
        live = torch.where(live_mask)[0]; old_removed = torch.where(removed_columns)[0]
        permutation = torch.cat([chosen_columns, live, old_removed])
        inverse = torch.argsort(permutation)
        Wp = W[:, permutation]
        Hp = Hinv[permutation][:, permutation]
        selected = chosen_columns.numel(); active_end = width - old_removed.numel()
        upper = torch.linalg.cholesky(Hp, upper=True)[:selected]
        selected_weight = Wp[:, :selected].clone(); errors_matrix = torch.zeros_like(selected_weight)
        for i in range(selected):
            errors_matrix[:, i:i + 1] = selected_weight[:, i:i + 1] / upper[i, i]
            selected_weight[:, i:] -= errors_matrix[:, i:i + 1].matmul(upper[i:i + 1, i:selected])
        Wp[:, :selected] = 0
        if active_end > selected:
            Wp[:, selected:active_end] -= errors_matrix.matmul(upper[:, selected:active_end])
        W = Wp[:, inverse]
        H[chosen_columns, :] = 0; H[:, chosen_columns] = 0; H[chosen_columns, chosen_columns] = 1
        removed_columns[chosen_columns] = True; removed_groups[chosen_groups] = True
        current += count_groups
        del Hinv, Hp, upper, selected_weight, errors_matrix
        torch.cuda.empty_cache()

    keep = torch.where(~removed_columns)[0]
    remove = torch.where(removed_columns)[0]
    return W[:, keep].to(weight.dtype).contiguous(), keep, remove


def linear(old: nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None):
    new = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                    device=old.weight.device, dtype=old.weight.dtype)
    new.weight.data.copy_(weight.to(old.weight.device, old.weight.dtype))
    if bias is not None:
        new.bias.data.copy_(bias.to(old.weight.device, old.weight.dtype))
    new.train(old.training)
    return new


def prune_target(a, unet, target: Target, H: torch.Tensor, ratio: float):
    block = by_path(unet, target.block_path)
    if target.kind == "ff":
        input_layer = block.ff.net[0].proj; output = block.ff.net[2]
        original_inner = output.in_features
        reduced, keep, removed = obs_remove_columns(
            output.weight, H, ratio, 1, a.percdamp, a.ff_prune_chunk
        )
        rows = torch.cat([keep, keep + original_inner])
        input_weight = input_layer.weight.index_select(0, rows)
        input_bias = input_layer.bias.index_select(0, rows) if input_layer.bias is not None else None
        block.ff.net[0].proj = linear(input_layer, input_weight, input_bias)
        block.ff.net[2] = linear(output, reduced, output.bias)
        return {"kind": "ff", "removed_neurons": int(removed.numel()),
                "original_inner_dim": original_inner, "new_inner_dim": int(keep.numel())}

    attn: Attention = getattr(block, target.attention_name)
    output = attn.to_out[0]
    reduced, keep, removed = obs_remove_columns(
        output.weight, H, ratio, target.group_size, a.percdamp, 1
    )
    attn.to_q = linear(attn.to_q, attn.to_q.weight.index_select(0, keep),
                       attn.to_q.bias.index_select(0, keep) if attn.to_q.bias is not None else None)
    attn.to_k = linear(attn.to_k, attn.to_k.weight.index_select(0, keep),
                       attn.to_k.bias.index_select(0, keep) if attn.to_k.bias is not None else None)
    attn.to_v = linear(attn.to_v, attn.to_v.weight.index_select(0, keep),
                       attn.to_v.bias.index_select(0, keep) if attn.to_v.bias is not None else None)
    attn.to_out[0] = linear(output, reduced, output.bias)
    original_heads = attn.heads; new_heads = keep.numel() // target.group_size
    attn.heads = int(new_heads); attn.inner_dim = int(keep.numel())
    attn.inner_kv_dim = int(keep.numel()); attn.sliceable_head_dim = int(new_heads)
    attn.fused_projections = False
    return {"kind": "attention", "attention": target.attention_name,
            "head_dim": target.group_size, "original_heads": original_heads,
            "new_heads": int(new_heads), "removed_heads": int(removed.numel() // target.group_size)}


def validate(unet, target: Target):
    block = by_path(unet, target.block_path)
    if target.kind == "ff":
        inner = block.ff.net[2].in_features
        if block.ff.net[0].proj.out_features != 2 * inner:
            raise RuntimeError(f"GEGLU mismatch at {target.target_id}")
    else:
        attn: Attention = getattr(block, target.attention_name)
        inner = attn.heads * target.group_size
        if not (attn.inner_dim == inner == attn.to_q.out_features == attn.to_k.out_features
                == attn.to_v.out_features == attn.to_out[0].in_features):
            raise RuntimeError(f"Attention mismatch at {target.target_id}")


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())

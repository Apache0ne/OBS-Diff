#!/usr/bin/env python3
"""Virtual-basis feed-forward compression utilities for SDXL.

The implementation replaces each SDXL GEGLU feed-forward network with a
narrower, newly trainable GEGLU basis. The compact basis is initialized from
high-contribution neurons of an OBS-zero teacher and then distilled against
cached teacher input/output activations. A second optional UNet-level replay
stage corrects distribution shift after many local replacements.

Only standard Diffusers FeedForward/GEGLU modules are installed into the final
UNet, so the exported whole-object .pth does not require this module at load
or inference time.
"""
from __future__ import annotations

import gc
import math
import random
import zlib
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import GEGLU
from diffusers.models.attention import FeedForward


@dataclass(frozen=True)
class FFNTarget:
    block_path: str
    ff_path: str
    input_dim: int
    inner_dim: int
    output_dim: int
    dropout: float
    final_dropout: bool
    bias: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def module_by_path(root: nn.Module, path: str) -> nn.Module:
    value: Any = root
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def discover_ffns(unet: nn.Module) -> "OrderedDict[str, FFNTarget]":
    result: "OrderedDict[str, FFNTarget]" = OrderedDict()
    for path, block in unet.named_modules():
        if ".transformer_blocks." not in f".{path}" or not hasattr(block, "ff"):
            continue
        ff = block.ff
        if not hasattr(ff, "net") or len(ff.net) < 3:
            continue
        if not isinstance(ff.net[0], GEGLU) or not isinstance(ff.net[2], nn.Linear):
            raise RuntimeError(f"Unsupported feed-forward structure at {path}.ff")
        proj = ff.net[0].proj
        down = ff.net[2]
        if not isinstance(proj, nn.Linear):
            raise RuntimeError(f"Unsupported GEGLU projection at {path}.ff")
        if proj.out_features != 2 * down.in_features:
            raise RuntimeError(f"GEGLU width mismatch at {path}.ff")
        if proj.in_features != down.out_features:
            raise RuntimeError(f"Residual width mismatch at {path}.ff")
        dropout = float(getattr(ff.net[1], "p", 0.0))
        final_dropout = len(ff.net) > 3 and isinstance(ff.net[-1], nn.Dropout)
        bias = proj.bias is not None
        if (down.bias is not None) != bias:
            raise RuntimeError(f"Mixed FFN bias configuration is unsupported at {path}.ff")
        result[path] = FFNTarget(
            block_path=path,
            ff_path=path + ".ff",
            input_dim=int(proj.in_features),
            inner_dim=int(down.in_features),
            output_dim=int(down.out_features),
            dropout=dropout,
            final_dropout=final_dropout,
            bias=bias,
        )
    if not result:
        raise RuntimeError("No SDXL GEGLU feed-forward targets were found")
    return result


def aligned_width(
    original: int,
    keep_fraction: float,
    multiple: int = 64,
    minimum: int = 256,
) -> int:
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction must satisfy 0 < keep_fraction <= 1")
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    raw = int(math.ceil(original * keep_fraction / multiple) * multiple)
    return min(original, max(minimum, raw))


def make_compact_ffn(
    old_ff: FeedForward,
    basis_dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> FeedForward:
    proj = old_ff.net[0].proj
    down = old_ff.net[2]
    if basis_dim <= 0 or basis_dim > down.in_features:
        raise ValueError("basis_dim is outside the valid FFN range")
    module = FeedForward(
        dim=int(proj.in_features),
        dim_out=int(down.out_features),
        mult=4,
        dropout=float(getattr(old_ff.net[1], "p", 0.0)),
        activation_fn="geglu",
        final_dropout=(len(old_ff.net) > 3 and isinstance(old_ff.net[-1], nn.Dropout)),
        inner_dim=int(basis_dim),
        bias=proj.bias is not None,
    )
    module.to(device=device, dtype=dtype)
    module.train()
    return module


def _stable_seed(text: str, base: int) -> int:
    return (int(base) + int(zlib.crc32(text.encode("utf-8")))) & 0x7FFFFFFF


class FFNActivationRecorder:
    """Sample paired FFN inputs and teacher outputs from many UNet calls."""

    def __init__(
        self,
        unet: nn.Module,
        targets: Mapping[str, FFNTarget],
        *,
        max_samples: int,
        tokens_per_call: int,
        seed: int,
        storage_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.unet = unet
        self.targets = targets
        self.max_samples = int(max_samples)
        self.tokens_per_call = int(tokens_per_call)
        self.seed = int(seed)
        self.storage_dtype = storage_dtype
        self.handles: List[Any] = []
        self.calls: Dict[str, int] = {path: 0 for path in targets}
        self.counts: Dict[str, int] = {path: 0 for path in targets}
        self.storage: Dict[str, Dict[str, List[torch.Tensor]]] = {
            path: {"x": [], "y": []} for path in targets
        }

    def _hook(self, path: str):
        def capture(module: nn.Module, inputs: Tuple[Any, ...], output: torch.Tensor):
            if self.counts[path] >= self.max_samples:
                self.calls[path] += 1
                return
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"FFN input was not a tensor at {path}")
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            y = output.detach().reshape(-1, output.shape[-1])
            if x.shape[0] != y.shape[0]:
                raise RuntimeError(f"FFN sample count mismatch at {path}")
            remaining = self.max_samples - self.counts[path]
            take = min(self.tokens_per_call, remaining, x.shape[0])
            call_index = self.calls[path]
            self.calls[path] += 1
            if take <= 0:
                return
            generator = torch.Generator(device=x.device)
            generator.manual_seed(_stable_seed(path, self.seed + 104729 * call_index))
            if take == x.shape[0]:
                indices = torch.arange(take, device=x.device)
            else:
                indices = torch.randperm(
                    x.shape[0], generator=generator, device=x.device
                )[:take]
            self.storage[path]["x"].append(
                x.index_select(0, indices)
                .to(device="cpu", dtype=self.storage_dtype)
                .contiguous()
            )
            self.storage[path]["y"].append(
                y.index_select(0, indices)
                .to(device="cpu", dtype=self.storage_dtype)
                .contiguous()
            )
            self.counts[path] += int(take)

        return capture

    def __enter__(self) -> "FFNActivationRecorder":
        for path in self.targets:
            block = module_by_path(self.unet, path)
            self.handles.append(block.ff.register_forward_hook(self._hook(path)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finalize(self) -> Dict[str, Dict[str, torch.Tensor]]:
        output: Dict[str, Dict[str, torch.Tensor]] = {}
        for path in self.targets:
            xs = self.storage[path]["x"]
            ys = self.storage[path]["y"]
            if not xs or not ys:
                raise RuntimeError(f"No FFN activation records were captured for {path}")
            x = torch.cat(xs, dim=0)[: self.max_samples].contiguous()
            y = torch.cat(ys, dim=0)[: self.max_samples].contiguous()
            if x.shape[0] != y.shape[0]:
                raise RuntimeError(f"Final FFN record mismatch at {path}")
            output[path] = {"x": x, "y": y}
        return output


def _clone_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone().contiguous()
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    return value


def move_nested(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_nested(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_nested(item, device) for item in value)
    if isinstance(value, list):
        return [move_nested(item, device) for item in value]
    return value


class UNetReplayRecorder:
    """Reservoir-sample complete UNet calls for final-output recovery."""

    def __init__(self, unet: nn.Module, max_records: int, seed: int) -> None:
        self.unet = unet
        self.max_records = int(max_records)
        self.random = random.Random(int(seed))
        self.records: List[Dict[str, Any]] = []
        self.seen = 0
        self.pending: Tuple[int | None, Dict[str, Any] | None] = (None, None)
        self.pre_handle = None
        self.post_handle = None

    def _pre(self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        self.seen += 1
        if len(self.records) < self.max_records:
            slot: int | None = len(self.records)
        else:
            candidate = self.random.randrange(self.seen)
            slot = candidate if candidate < self.max_records else None
        payload = None
        if slot is not None:
            payload = {"args": _clone_cpu(args), "kwargs": _clone_cpu(kwargs)}
        self.pending = (slot, payload)

    def _post(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        output: Any,
    ):
        slot, payload = self.pending
        self.pending = (None, None)
        if slot is None or payload is None:
            return
        sample = (
            output.sample
            if hasattr(output, "sample")
            else output[0]
            if isinstance(output, tuple)
            else output
        )
        payload["target"] = _clone_cpu(sample)
        if slot == len(self.records):
            self.records.append(payload)
        else:
            self.records[slot] = payload

    def __enter__(self) -> "UNetReplayRecorder":
        self.pre_handle = self.unet.register_forward_pre_hook(
            self._pre, with_kwargs=True
        )
        self.post_handle = self.unet.register_forward_hook(
            self._post, with_kwargs=True
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.pre_handle is not None:
            self.pre_handle.remove()
        if self.post_handle is not None:
            self.post_handle.remove()
        self.pre_handle = None
        self.post_handle = None


def teacher_neuron_scores(
    old_ff: FeedForward,
    x_cpu: torch.Tensor,
    *,
    device: torch.device | str,
    batch_size: int,
) -> torch.Tensor:
    model_dtype = old_ff.net[0].proj.weight.dtype
    old_ff = old_ff.to(device=device)
    old_ff.eval()
    inner = old_ff.net[2].in_features
    energy = torch.zeros(inner, dtype=torch.float64, device="cpu")
    count = 0
    with torch.inference_mode():
        for start in range(0, x_cpu.shape[0], batch_size):
            x = x_cpu[start : start + batch_size].to(
                device=device, dtype=model_dtype
            )
            z = old_ff.net[0](x)
            energy.add_(z.float().square().sum(dim=0).double().cpu())
            count += int(z.shape[0])
    energy.div_(max(count, 1))
    output_norm = (
        old_ff.net[2]
        .weight.detach()
        .float()
        .square()
        .sum(dim=0)
        .double()
        .cpu()
    )
    score = energy * output_norm
    old_ff.to("cpu")
    torch.cuda.empty_cache()
    return score.float()


def initialize_compact_from_teacher(
    old_ff: FeedForward,
    compact: FeedForward,
    keep: torch.Tensor,
) -> None:
    keep = keep.to(dtype=torch.long, device="cpu")
    old_proj = old_ff.net[0].proj
    old_down = old_ff.net[2]
    old_inner = old_down.in_features
    rows = torch.cat([keep, keep + old_inner], dim=0)
    compact.net[0].proj.weight.data.copy_(
        old_proj.weight.detach().cpu().index_select(0, rows).to(
            device=compact.net[0].proj.weight.device,
            dtype=compact.net[0].proj.weight.dtype,
        )
    )
    if compact.net[0].proj.bias is not None:
        compact.net[0].proj.bias.data.copy_(
            old_proj.bias.detach().cpu().index_select(0, rows).to(
                device=compact.net[0].proj.bias.device,
                dtype=compact.net[0].proj.bias.dtype,
            )
        )
    compact.net[2].weight.data.copy_(
        old_down.weight.detach().cpu().index_select(1, keep).to(
            device=compact.net[2].weight.device,
            dtype=compact.net[2].weight.dtype,
        )
    )
    if compact.net[2].bias is not None:
        compact.net[2].bias.data.copy_(
            old_down.bias.detach().to(
                device=compact.net[2].bias.device,
                dtype=compact.net[2].bias.dtype,
            )
        )


def _normalized_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    mse = F.mse_loss(prediction, target)
    scale = target.square().mean().clamp_min(1e-6)
    normalized = mse / scale
    if cosine_weight <= 0:
        return normalized
    cosine = 1.0 - F.cosine_similarity(
        prediction, target, dim=-1, eps=1e-6
    ).mean()
    return normalized + float(cosine_weight) * cosine


@torch.no_grad()
def evaluate_ffn(
    module: FeedForward,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device | str,
    batch_size: int,
    amp_dtype: torch.dtype,
    cosine_weight: float,
) -> float:
    module.eval()
    total = 0.0
    count = 0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(
            device=device, dtype=torch.float32
        )
        yb = y[start : start + batch_size].to(
            device=device, dtype=torch.float32
        )
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            pred = module(xb)
        loss = _normalized_loss(pred, yb, cosine_weight)
        batch_count = int(xb.shape[0])
        total += float(loss) * batch_count
        count += batch_count
    module.train()
    return total / max(count, 1)


def distill_compact_ffn(
    old_ff: FeedForward,
    records: Mapping[str, torch.Tensor],
    *,
    basis_dim: int,
    device: torch.device | str,
    model_dtype: torch.dtype,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    cosine_weight: float,
    validation_fraction: float,
    seed: int,
    log_every: int,
) -> Tuple[FeedForward, Dict[str, Any]]:
    x = records["x"].contiguous()
    y = records["y"].contiguous()
    if x.shape[0] != y.shape[0] or x.shape[0] < 16:
        raise RuntimeError("Insufficient paired FFN records")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(x.shape[0], generator=generator)
    validation_count = max(8, int(round(x.shape[0] * validation_fraction)))
    validation_count = min(validation_count, x.shape[0] // 3)
    val_idx = permutation[:validation_count]
    train_idx = permutation[validation_count:]
    x_train = x.index_select(0, train_idx)
    y_train = y.index_select(0, train_idx)
    x_val = x.index_select(0, val_idx)
    y_val = y.index_select(0, val_idx)

    score = teacher_neuron_scores(
        old_ff, x_train, device=device, batch_size=batch_size
    )
    keep = (
        torch.topk(score, k=int(basis_dim), largest=True, sorted=True)
        .indices.sort()
        .values
    )
    compact = make_compact_ffn(
        old_ff, basis_dim, device=device, dtype=torch.float32
    )
    initialize_compact_from_teacher(old_ff, compact, keep)

    amp_dtype = (
        torch.float16 if model_dtype == torch.float16 else torch.bfloat16
    )
    optimizer = torch.optim.AdamW(
        compact.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1)
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(amp_dtype == torch.float16)
    )

    initial_val = evaluate_ffn(
        compact,
        x_val,
        y_val,
        device=device,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
        cosine_weight=cosine_weight,
    )
    best_val = initial_val
    best_step = 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in compact.state_dict().items()
    }
    losses: List[float] = []

    compact.train()
    for step in range(int(steps)):
        batch_generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + 1009 * (step + 1)
        )
        if batch_size >= x_train.shape[0]:
            idx = torch.arange(x_train.shape[0])
        else:
            idx = torch.randint(
                0,
                x_train.shape[0],
                (batch_size,),
                generator=batch_generator,
            )
        xb = x_train.index_select(0, idx).to(
            device=device, dtype=torch.float32
        )
        yb = y_train.index_select(0, idx).to(
            device=device, dtype=torch.float32
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            prediction = compact(xb)
            loss = _normalized_loss(prediction, yb, cosine_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(compact.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(float(loss.detach()))

        should_validate = (
            (step + 1) % max(log_every, 1) == 0 or step + 1 == steps
        )
        if should_validate:
            value = evaluate_ffn(
                compact,
                x_val,
                y_val,
                device=device,
                batch_size=batch_size,
                amp_dtype=amp_dtype,
                cosine_weight=cosine_weight,
            )
            print(
                f"      step {step + 1:4d}/{steps} "
                f"train={losses[-1]:.6f} val={value:.6f}"
            )
            if value < best_val:
                best_val = value
                best_step = step + 1
                best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in compact.state_dict().items()
                }

    compact.load_state_dict(best_state)
    compact.eval()
    compact.to(device="cpu", dtype=model_dtype)
    for parameter in compact.parameters():
        parameter.requires_grad_(False)
    del optimizer, scheduler, scaler, best_state
    torch.cuda.empty_cache()
    stats = {
        "basis_dim": int(basis_dim),
        "selected_teacher_neurons": keep.tolist(),
        "initial_validation_loss": float(initial_val),
        "best_validation_loss": float(best_val),
        "best_step": int(best_step),
        "final_training_loss": float(losses[-1]) if losses else None,
        "steps": int(steps),
        "samples": int(x.shape[0]),
    }
    return compact, stats


def clear_gradient_checkpointing_state(unet: nn.Module) -> Dict[str, int]:
    try:
        unet.disable_gradient_checkpointing()
    except Exception:
        pass
    disabled = 0
    removed = 0
    for module in unet.modules():
        if hasattr(module, "gradient_checkpointing"):
            try:
                module.gradient_checkpointing = False
                disabled += 1
            except Exception:
                pass
        if "_gradient_checkpointing_func" in module.__dict__:
            del module.__dict__["_gradient_checkpointing_func"]
            removed += 1
    return {
        "disabled_modules": disabled,
        "removed_runtime_closures": removed,
    }


def global_recover_student(
    unet: nn.Module,
    replay_records: Sequence[Mapping[str, Any]],
    compressed_paths: Sequence[str],
    *,
    device: torch.device | str,
    model_dtype: torch.dtype,
    group_size: int,
    steps_per_group: int,
    learning_rate: float,
    cosine_weight: float,
) -> List[Dict[str, Any]]:
    if not replay_records or steps_per_group <= 0 or not compressed_paths:
        cleanup = clear_gradient_checkpointing_state(unet)
        print(f"  global recovery skipped; checkpoint cleanup={cleanup}")
        return []

    unet.to(device=device, dtype=model_dtype)
    unet.train()
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    try:
        unet.enable_gradient_checkpointing()
    except Exception:
        pass

    amp_dtype = (
        torch.float16 if model_dtype == torch.float16 else torch.bfloat16
    )
    histories: List[Dict[str, Any]] = []
    paths = list(compressed_paths)
    effective_group = max(group_size, 1)
    for group_index, start in enumerate(range(0, len(paths), effective_group)):
        group = paths[start : start + effective_group]
        trainable: List[nn.Parameter] = []
        for path in group:
            ff = module_by_path(unet, path).ff
            for parameter in ff.parameters():
                parameter.data = parameter.data.float()
                parameter.requires_grad_(True)
                trainable.append(parameter)
        optimizer = torch.optim.AdamW(
            trainable, lr=float(learning_rate), weight_decay=0.0
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(amp_dtype == torch.float16)
        )
        group_losses: List[float] = []
        print(f"  global group {group_index + 1}: {len(group)} FFNs")
        try:
            for step in range(int(steps_per_group)):
                record = replay_records[
                    (group_index * steps_per_group + step)
                    % len(replay_records)
                ]
                args = move_nested(record["args"], device)
                kwargs = move_nested(record["kwargs"], device)
                target = move_nested(record["target"], device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    output = unet(*args, **kwargs)
                    prediction = (
                        output.sample if hasattr(output, "sample") else output[0]
                    )
                    loss = _normalized_loss(
                        prediction, target, cosine_weight
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                group_losses.append(float(loss.detach()))
                print(
                    f"    step {step + 1}/{steps_per_group}: "
                    f"loss={group_losses[-1]:.6f}"
                )
        finally:
            optimizer.zero_grad(set_to_none=True)
            for parameter in trainable:
                parameter.requires_grad_(False)
                parameter.data = parameter.data.to(dtype=model_dtype)
            del optimizer, scaler, trainable
            torch.cuda.empty_cache()
        histories.append({"paths": group, "losses": group_losses})

    unet.eval()
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    cleanup = clear_gradient_checkpointing_state(unet)
    print(f"  global recovery checkpoint cleanup={cleanup}")
    unet.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return histories


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def count_parameter_bytes(module: nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )

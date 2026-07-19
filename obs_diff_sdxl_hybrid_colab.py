#!/usr/bin/env python3
"""L4-safe entrypoint for the static Diff-ES + OBS-Diff hybrid runner."""
from __future__ import annotations

import math
import torch

import obs_diff_sdxl_hybrid as core


def expanded_ratios(target):
    # FFNs remain the primary source of physical reduction. The aggressive tail
    # exists so 40/50% whole-UNet experiments are reachable; the OBS cost and
    # attention penalties keep these levels unused unless the budget requires it.
    if target.kind == "ff":
        return [
            0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95,
        ]
    heads = target.width // target.group_size
    # Cross-attention remains more restricted than self-attention. These upper
    # limits are primarily for the extreme 40/50% whole-UNet experiments.
    maximum = 0.30 if target.attention_name == "attn2" else 0.50
    max_remove = max(1, int(math.floor(heads * maximum)))
    return [k / heads for k in range(0, max_remove + 1)]


def l4_recover_student(a, unet, record_path, plan):
    """Output-projection teacher recovery with zero optimizer-state overhead."""
    if a.recovery_steps <= 0 or record_path is None:
        return []
    records = torch.load(record_path, map_location="cpu", weights_only=False)
    for parameter in unet.parameters():
        parameter.requires_grad_(False)

    trainable = []
    seen = set()
    for target_id, selected in plan.items():
        if selected["ratio"] <= 0:
            continue
        block_path = target_id.rsplit(".", 1)[0]
        block = core.by_path(unet, block_path)
        if selected["kind"] == "ff":
            module = block.ff.net[2]
        else:
            module = getattr(block, selected["attention_name"]).to_out[0]
        for parameter in module.parameters():
            if id(parameter) not in seen:
                parameter.requires_grad_(True)
                trainable.append(parameter)
                seen.add(id(parameter))

    if not trainable:
        return []

    unet.train()
    optimizer = torch.optim.SGD(trainable, lr=a.recovery_lr, momentum=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=a.dtype == "float16")
    losses = []

    for step in range(a.recovery_steps):
        record = records[step % len(records)]
        optimizer.zero_grad(set_to_none=True)
        dtype = next(unet.parameters()).dtype
        sample = record["sample"].to("cuda", dtype=dtype)
        timestep = record["t"].to("cuda")
        encoder_hidden_states = record["encoder_hidden_states"].to("cuda", dtype=dtype)
        text_embeds = record["text_embeds"].to("cuda", dtype=dtype)
        time_ids = record["time_ids"].to("cuda", dtype=dtype)
        teacher_target = record["target"].to("cuda", dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            prediction = unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample
            loss = torch.nn.functional.mse_loss(prediction.float(), teacher_target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))
        print(f"  recovery step {step + 1:03d}/{a.recovery_steps}: loss={losses[-1]:.8f}")

        del sample, encoder_hidden_states, text_embeds, time_ids, teacher_target, prediction, loss
        torch.cuda.empty_cache()

    unet.eval()
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    return losses


core.allowed_ratios = expanded_ratios
core.recover_student = l4_recover_student

if __name__ == "__main__":
    core.main()

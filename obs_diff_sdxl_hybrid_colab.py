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


def clear_gradient_checkpointing_state(unet):
    """Remove Diffusers runtime checkpoint closures so the UNet is pickle-safe."""
    try:
        unet.disable_gradient_checkpointing()
    except Exception:
        pass

    removed_closures = 0
    disabled_modules = 0
    for module in unet.modules():
        if hasattr(module, "gradient_checkpointing"):
            try:
                module.gradient_checkpointing = False
                disabled_modules += 1
            except Exception:
                pass
        if "_gradient_checkpointing_func" in module.__dict__:
            del module.__dict__["_gradient_checkpointing_func"]
            removed_closures += 1

    print(
        "  gradient checkpoint cleanup: "
        f"disabled_modules={disabled_modules} "
        f"removed_runtime_closures={removed_closures}"
    )


def l4_recover_student(a, unet, record_path, plan):
    """Recover selected output projections with FP32 parameter gradients.

    GradScaler cannot unscale gradients stored directly in FP16 parameters. The
    physically pruned UNet remains FP16, but selected output projections are
    temporarily stored in FP32 for recovery, then cast back before export.
    """
    if a.recovery_steps <= 0 or record_path is None:
        clear_gradient_checkpointing_state(unet)
        return []

    records = torch.load(record_path, map_location="cpu", weights_only=False)
    if not records:
        clear_gradient_checkpointing_state(unet)
        return []

    model_dtype = next(unet.parameters()).dtype
    for parameter in unet.parameters():
        parameter.requires_grad_(False)

    trainable_modules = []
    trainable = []
    seen_modules = set()
    seen_parameters = set()

    for target_id, selected in plan.items():
        if selected["ratio"] <= 0:
            continue
        block_path = target_id.rsplit(".", 1)[0]
        block = core.by_path(unet, block_path)
        if selected["kind"] == "ff":
            module = block.ff.net[2]
        else:
            module = getattr(block, selected["attention_name"]).to_out[0]

        if id(module) not in seen_modules:
            module.to(dtype=torch.float32)
            trainable_modules.append(module)
            seen_modules.add(id(module))

        for parameter in module.parameters():
            if id(parameter) not in seen_parameters:
                parameter.requires_grad_(True)
                trainable.append(parameter)
                seen_parameters.add(id(parameter))

    if not trainable:
        clear_gradient_checkpointing_state(unet)
        return []

    unet.enable_gradient_checkpointing()
    unet.train()
    optimizer = torch.optim.SGD(trainable, lr=a.recovery_lr, momentum=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(model_dtype == torch.float16))
    losses = []

    try:
        for step in range(a.recovery_steps):
            record = records[step % len(records)]
            optimizer.zero_grad(set_to_none=True)

            sample = record["sample"].to("cuda", dtype=model_dtype)
            timestep = record["t"].to("cuda")
            encoder_hidden_states = record["encoder_hidden_states"].to(
                "cuda", dtype=model_dtype
            )
            text_embeds = record["text_embeds"].to("cuda", dtype=model_dtype)
            time_ids = record["time_ids"].to("cuda", dtype=model_dtype)
            teacher_target = record["target"].to("cuda", dtype=torch.float32)

            with torch.autocast(
                device_type="cuda",
                dtype=model_dtype,
                enabled=model_dtype in (torch.float16, torch.bfloat16),
            ):
                prediction = unet(
                    sample,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": text_embeds,
                        "time_ids": time_ids,
                    },
                ).sample
                loss = torch.nn.functional.mse_loss(
                    prediction.float(), teacher_target
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.item()))
            print(
                f"  recovery step {step + 1:03d}/{a.recovery_steps}: "
                f"loss={losses[-1]:.8f}"
            )

            del (
                sample,
                timestep,
                encoder_hidden_states,
                text_embeds,
                time_ids,
                teacher_target,
                prediction,
                loss,
            )
            torch.cuda.empty_cache()
    finally:
        for module in trainable_modules:
            module.to(dtype=model_dtype)
        unet.eval()
        for parameter in unet.parameters():
            parameter.requires_grad_(False)
        clear_gradient_checkpointing_state(unet)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler, records
        torch.cuda.empty_cache()

    return losses


core.allowed_ratios = expanded_ratios
core.recover_student = l4_recover_student
core.clear_gradient_checkpointing_state = clear_gradient_checkpointing_state

if __name__ == "__main__":
    core.main()

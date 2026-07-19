#!/usr/bin/env python3
"""CPU structural smoke test for the SDXL virtual-basis implementation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward

from lib.virtual_basis_sdxl import (
    FFNActivationRecorder,
    UNetReplayRecorder,
    aligned_width,
    count_parameters,
    discover_ffns,
    initialize_compact_from_teacher,
    make_compact_ffn,
)


class MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = FeedForward(64, inner_dim=256, activation_fn="geglu")

    def forward(self, hidden_states):
        return hidden_states + self.ff(hidden_states)


class MockTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([MockBlock(), MockBlock()])

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class MockOutput:
    def __init__(self, sample):
        self.sample = sample


class MockUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = MockTransformer()

    def forward(
        self,
        sample,
        timestep=None,
        encoder_hidden_states=None,
        added_cond_kwargs=None,
    ):
        return MockOutput(self.down(sample))


def main():
    unet = MockUNet()
    targets = discover_ffns(unet)
    if len(targets) != 2:
        raise RuntimeError(f"Expected two mock FFNs, found {len(targets)}")

    hidden = torch.randn(2, 12, 64)
    with FFNActivationRecorder(
        unet,
        targets,
        max_samples=16,
        tokens_per_call=8,
        seed=7,
    ) as recorder:
        unet(hidden)
        unet(hidden)
    records = recorder.finalize()
    if not all(value["x"].shape == (16, 64) for value in records.values()):
        raise RuntimeError("FFN activation record shape test failed")

    with UNetReplayRecorder(unet, max_records=2, seed=7) as replay:
        for step in range(4):
            unet(
                hidden,
                torch.tensor(step),
                encoder_hidden_states=torch.randn(1, 77, 64),
                added_cond_kwargs={"text_embeds": torch.randn(1, 32)},
            )
    if len(replay.records) != 2:
        raise RuntimeError("UNet replay reservoir test failed")

    old_ff = unet.down.transformer_blocks[0].ff
    compact = make_compact_ffn(
        old_ff,
        128,
        device="cpu",
        dtype=torch.float32,
    )
    initialize_compact_from_teacher(old_ff, compact, torch.arange(128))
    unet.down.transformer_blocks[0].ff = compact

    if unet(hidden).sample.shape != hidden.shape:
        raise RuntimeError("Compact FFN forward shape test failed")
    if count_parameters(compact) >= count_parameters(old_ff):
        raise RuntimeError("Compact FFN did not reduce parameters")
    if aligned_width(5120, 0.65, 64, 256) != 3328:
        raise RuntimeError("Width alignment test failed")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mock_virtual_basis_unet.pth"
        torch.save(unet, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if loaded(hidden).sample.shape != hidden.shape:
            raise RuntimeError("Whole-object reload test failed")

    print("VIRTUAL BASIS SDXL SELF-TEST: PASS")


if __name__ == "__main__":
    main()

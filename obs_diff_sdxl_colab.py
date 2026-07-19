#!/usr/bin/env python3
"""Memory-safe Colab entrypoint for obs_diff_sdxl.

The core runner calibrates each Hessian package once. This wrapper replaces its
package writer so each pruned module is serialized immediately instead of
holding four package-sized ratio dictionaries in CPU memory.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from safetensors.torch import save_file

import obs_diff_sdxl as core


def memory_safe_make_shards(a, root: Path, package_id: int, names, modules, stats, rs):
    for module_id, name in enumerate(names):
        dense = modules[name].weight.detach().cpu().contiguous()
        print(f"  OBS {module_id + 1}/{len(names)} {name} {tuple(dense.shape)}")
        Hinv, dead = core.hessian_inverse(stats[name], a.percdamp)

        for ratio in rs:
            started = time.perf_counter()
            weight = core.obs_weight(
                dense=dense,
                Hinv=Hinv,
                dead=dead,
                ratio=ratio,
                block=a.column_block,
            )
            folder = root / "shards" / f"sparsity_{core.label(ratio)}"
            folder.mkdir(parents=True, exist_ok=True)
            shard = folder / f"package_{package_id:03d}_module_{module_id:04d}.safetensors"
            save_file(
                {name + ".weight": weight},
                str(shard),
                metadata={
                    "method": "OBS-Diff",
                    "model_family": "SDXL",
                    "ratio": str(ratio),
                    "module": name,
                },
            )
            zero_fraction = float((weight == 0).sum().item() / weight.numel())
            print(
                f"    {ratio:.0%}: zeros={zero_fraction:.6f}, "
                f"saved={shard.name}, time={time.perf_counter() - started:.2f}s"
            )
            del weight

        del dense, Hinv, dead
        torch.cuda.empty_cache()


core.make_shards = memory_safe_make_shards

if __name__ == "__main__":
    core.main()

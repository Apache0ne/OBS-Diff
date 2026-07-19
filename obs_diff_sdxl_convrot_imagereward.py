#!/usr/bin/env python3
"""ImageReward scorer for structured SDXL ConvRot comparisons.

Run in an isolated Transformers 4.36.2 child process, matching the working
OBS-Diff inference-only ImageReward path. No SDXL model is loaded here.
"""
from __future__ import annotations

import argparse
import base64
import html
import importlib
import importlib.machinery
import json
import os
import sys
import types
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-html", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--med-config", required=True)
    p.add_argument("--compat-root", required=True)
    return p.parse_args()


def prepare(root: Path):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "DIFFUSERS_OFFLINE"):
        os.environ.pop(key, None)
    import transformers
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu

    loaded = Path(transformers.__file__).resolve()
    if root not in loaded.parents or transformers.__version__ != "4.36.2":
        raise RuntimeError(
            f"Expected Transformers 4.36.2 under {root}; loaded {transformers.__version__} from {loaded}"
        )
    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(mu, name):
            if not hasattr(pu, name):
                raise RuntimeError(f"Missing legacy utility {name}")
            setattr(mu, name, getattr(pu, name))


def import_inference():
    source = None
    for entry in sys.path:
        candidate = Path(entry) / "ImageReward"
        if (candidate / "utils.py").is_file() and (candidate / "ImageReward.py").is_file():
            source = candidate.resolve()
            break
    if source is None:
        raise RuntimeError("ImageReward package not found")
    for name in list(sys.modules):
        if name == "ImageReward" or name.startswith("ImageReward."):
            del sys.modules[name]
    package = types.ModuleType("ImageReward")
    package.__file__ = str(source / "__init__.py")
    package.__package__ = "ImageReward"
    package.__path__ = [str(source)]
    spec = importlib.machinery.ModuleSpec("ImageReward", loader=None, is_package=True)
    spec.submodule_search_locations = [str(source)]
    package.__spec__ = spec
    sys.modules["ImageReward"] = package
    utils = importlib.import_module("ImageReward.utils")
    for forbidden in ("ImageReward.ReFL", "datasets", "diffusers"):
        if forbidden in sys.modules:
            raise RuntimeError(f"Unexpected inference import: {forbidden}")
    return utils, source


def score_value(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.detach().float().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    return float(value)


def uri(path):
    buffer = BytesIO()
    Image.open(path).convert("RGB").save(buffer, "JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def build(data: Dict[str, Any], path: Path):
    variants = list(data["variants"])
    summary = data["imagereward_summary"]
    css = (
        "body{font-family:Arial;background:#f3f4f6;margin:20px}.card{background:white;padding:15px;"
        "margin:12px 0;border:1px solid #ccc}.grid{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));"
        "gap:8px;overflow-x:auto}.win{border-left:6px solid #238636}.loss{border-left:6px solid #b42318}"
        "img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #bbb;"
        "padding:6px}pre{white-space:pre-wrap;background:#111;color:#eee;padding:10px}"
    )
    out = [
        f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
        "<h1>Structured OBS-Diff SDXL + Comfy ConvRot ImageReward comparison</h1>",
        "<section class='card'><table><tr><th>Variant</th><th>Mean IR</th><th>Mean delta</th>"
        "<th>Median delta</th><th>Wins</th><th>File GiB</th><th>Actual storage GiB</th>"
        "<th>Quantized coverage</th><th>Mean seconds</th><th>Peak VRAM GiB</th></tr>",
    ]
    for variant in variants:
        reward = summary[variant]
        model = data["variants"][variant]
        out.append(
            f"<tr><td>{html.escape(variant)}</td><td>{reward['mean_imagereward']:+.6f}</td>"
            f"<td>{reward['mean_delta_vs_dense']:+.6f}</td>"
            f"<td>{reward['median_delta_vs_dense']:+.6f}</td>"
            f"<td>{reward['wins_vs_dense']}/{reward['cases']}</td>"
            f"<td>{model.get('pth_bytes',0)/1024**3:.3f}</td>"
            f"<td>{model.get('actual_model_storage_bytes',0)/1024**3:.3f}</td>"
            f"<td>{100*model.get('quantized_fraction_of_unet_parameters',0):.3f}%</td>"
            f"<td>{model['mean_generation_seconds']:.4f}</td>"
            f"<td>{model.get('peak_vram_bytes',0)/1024**3:.3f}</td></tr>"
        )
    out.append("</table></section>")
    for case in data["cases"]:
        dense = float(case["images"]["dense"]["imagereward"])
        out.append(f"<section class='card'><h2>{html.escape(case['prompt'])}</h2><div class='grid'>")
        for variant in variants:
            record = case["images"][variant]
            score = float(record["imagereward"])
            delta = score - dense
            cls = "win" if delta > 0 else "loss" if delta < 0 else ""
            detail = f"IR {score:+.5f} | delta {delta:+.5f} | {record['generation_seconds']:.3f}s"
            if variant != "dense":
                metric = case["metrics"][variant]
                detail += f" | PSNR {metric['psnr_db']:.2f} | MAE {metric['mae_0_1']:.4f}"
            out.append(
                f"<figure class='{cls}'><b>{html.escape(variant)}</b><br><small>{detail}</small>"
                f"<img src='{uri(record['path'])}'></figure>"
            )
        out.append("</div></section>")
    out.append(
        "<section class='card'><p>The reference is the physically structured FP16 UNet. ConvRot variants remain "
        "complete standalone structured UNets and use comfy-kitchen INT8/W4A4 tensor-core kernels.</p></section>"
    )
    out.append("<section class='card'><pre>" + html.escape(json.dumps(data, indent=2)) + "</pre></section></body></html>")
    path.write_text("".join(out), encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.compat_root).resolve()
    prepare(root)
    utils, source = import_inference()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = utils.load(
        name=str(Path(args.checkpoint).resolve()),
        device=device,
        med_config=str(Path(args.med_config).resolve()),
    )
    variants = list(data["variants"])
    for index, case in enumerate(data["cases"], 1):
        print(f"Case {index}/{len(data['cases'])}: {case['prompt']}")
        for variant in variants:
            with torch.inference_mode():
                value = score_value(model.score(case["prompt"], case["images"][variant]["path"]))
            case["images"][variant]["imagereward"] = value
            print(f"  {variant}: {value:+.6f}")
    dense = [float(case["images"]["dense"]["imagereward"]) for case in data["cases"]]
    summary = {}
    for variant in variants:
        scores = [float(case["images"][variant]["imagereward"]) for case in data["cases"]]
        deltas = [score - reference for score, reference in zip(scores, dense)]
        summary[variant] = {
            "cases": len(scores),
            "mean_imagereward": float(np.mean(scores)),
            "mean_delta_vs_dense": float(np.mean(deltas)),
            "median_delta_vs_dense": float(np.median(deltas)),
            "minimum_delta_vs_dense": float(np.min(deltas)),
            "maximum_delta_vs_dense": float(np.max(deltas)),
            "wins_vs_dense": int(sum(delta > 0 for delta in deltas)),
            "losses_vs_dense": int(sum(delta < 0 for delta in deltas)),
        }
    data["imagereward_summary"] = summary
    data["imagereward_environment"] = {
        "transformers": __import__("transformers").__version__,
        "image_reward_source": str(source),
        "refl_imported": "ImageReward.ReFL" in sys.modules,
        "datasets_imported": "datasets" in sys.modules,
        "diffusers_imported": "diffusers" in sys.modules,
    }
    Path(args.output_json).write_text(json.dumps(data, indent=2), encoding="utf-8")
    build(data, Path(args.output_html))
    print(f"Scored JSON: {args.output_json}\nScored HTML: {args.output_html}")


if __name__ == "__main__":
    main()

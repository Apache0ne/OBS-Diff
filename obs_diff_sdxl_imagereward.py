#!/usr/bin/env python3
"""Score OBS-Diff SDXL comparison images with ImageReward.

Run in a child process whose PYTHONPATH starts with a Transformers 4.36.2
--target directory. The ImageReward package root is bypassed so its ReFL,
datasets, and Diffusers training imports are not executed.
"""
from __future__ import annotations

import argparse, base64, html, importlib, importlib.machinery, json, os, sys, types
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--input-json",required=True)
    p.add_argument("--output-json",required=True)
    p.add_argument("--output-html",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--med-config",required=True)
    p.add_argument("--compat-root",required=True)
    return p.parse_args()


def prepare(root: Path):
    for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_DATASETS_OFFLINE","DIFFUSERS_OFFLINE"):
        os.environ.pop(key,None)
    import transformers
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu
    loaded=Path(transformers.__file__).resolve()
    if root not in loaded.parents or transformers.__version__!="4.36.2":
        raise RuntimeError(f"Expected Transformers 4.36.2 under {root}; loaded {transformers.__version__} from {loaded}")
    for name in ("apply_chunking_to_forward","find_pruneable_heads_and_indices","prune_linear_layer"):
        if not hasattr(mu,name):
            if not hasattr(pu,name): raise RuntimeError(f"Missing legacy utility {name}")
            setattr(mu,name,getattr(pu,name))


def import_inference():
    source=None
    for entry in sys.path:
        candidate=Path(entry)/"ImageReward"
        if (candidate/"utils.py").is_file() and (candidate/"ImageReward.py").is_file():
            source=candidate.resolve(); break
    if source is None: raise RuntimeError("ImageReward package not found")
    for name in list(sys.modules):
        if name=="ImageReward" or name.startswith("ImageReward."): del sys.modules[name]
    package=types.ModuleType("ImageReward"); package.__file__=str(source/"__init__.py")
    package.__package__="ImageReward"; package.__path__=[str(source)]
    spec=importlib.machinery.ModuleSpec("ImageReward",loader=None,is_package=True)
    spec.submodule_search_locations=[str(source)]; package.__spec__=spec
    sys.modules["ImageReward"]=package
    utils=importlib.import_module("ImageReward.utils")
    for forbidden in ("ImageReward.ReFL","datasets","diffusers"):
        if forbidden in sys.modules: raise RuntimeError(f"Unexpected inference import: {forbidden}")
    return utils,source


def score_value(value: Any)->float:
    if isinstance(value,(list,tuple)): value=value[0]
    if isinstance(value,torch.Tensor): value=value.detach().float().reshape(-1)[0].item()
    if isinstance(value,np.ndarray): value=value.reshape(-1)[0].item()
    return float(value)


def uri(path):
    buf=BytesIO(); Image.open(path).convert("RGB").save(buf,"JPEG",quality=90,optimize=True)
    return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()


def build(data: Dict[str,Any], path: Path):
    variants=list(data["variants"].keys()); summary=data["imagereward_summary"]
    structured=any("parameter_reduction_fraction" in data["variants"][v] for v in variants)
    css="body{font-family:Arial;background:#f3f4f6;margin:20px}.card{background:white;padding:15px;margin:12px 0;border:1px solid #ccc}.grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:8px;overflow-x:auto}.win{border-left:6px solid #238636}.loss{border-left:6px solid #b42318}img{width:100%}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:6px}pre{white-space:pre-wrap;background:#111;color:#eee;padding:10px}"
    title="Structured OBS-Diff SDXL ImageReward comparison" if structured else "OBS-Diff SDXL ImageReward comparison"
    out=[f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body><h1>{title}</h1>"]
    if structured:
        out.append("<section class='card'><table><tr><th>Variant</th><th>Mean IR</th><th>Mean delta</th><th>Median delta</th><th>Wins</th><th>Physical parameter reduction</th><th>UNet params</th><th>.pth GiB</th><th>Mean seconds</th></tr>")
    else:
        out.append("<section class='card'><table><tr><th>Variant</th><th>Mean IR</th><th>Mean delta</th><th>Median delta</th><th>Wins</th><th>Target sparsity</th><th>Overall UNet zeros</th><th>Mean seconds</th></tr>")
    for v in variants:
        s=summary[v]; p=data["variants"][v]
        prefix=f"<tr><td>{v}</td><td>{s['mean_imagereward']:+.6f}</td><td>{s['mean_delta_vs_dense']:+.6f}</td><td>{s['median_delta_vs_dense']:+.6f}</td><td>{s['wins_vs_dense']}/{s['cases']}</td>"
        if structured:
            row=prefix+f"<td>{100*p.get('parameter_reduction_fraction',0):.4f}%</td><td>{p.get('unet_parameters',0)/1e9:.4f}B</td><td>{p.get('pth_bytes',0)/1024**3:.3f}</td><td>{p['mean_generation_seconds']:.4f}</td></tr>"
        else:
            row=prefix+f"<td>{100*p.get('target_sparsity',0):.4f}%</td><td>{100*p.get('overall_unet_zero_fraction',0):.4f}%</td><td>{p['mean_generation_seconds']:.4f}</td></tr>"
        out.append(row)
    out.append("</table></section>")
    for case in data["cases"]:
        dense=float(case["images"]["dense"]["imagereward"])
        out.append(f"<section class='card'><h2>{html.escape(case['prompt'])}</h2><div class='grid'>")
        for v in variants:
            rec=case["images"][v]; score=float(rec["imagereward"]); delta=score-dense
            cls="win" if delta>0 else "loss" if delta<0 else ""
            detail=f"IR {score:+.5f} | delta {delta:+.5f} | {rec['generation_seconds']:.3f}s"
            if v!="dense":
                m=case["metrics"][v]; detail+=f" | PSNR {m['psnr_db']:.2f} | MAE {m['mae_0_1']:.4f}"
            out.append(f"<figure class='{cls}'><b>{v}</b><br><small>{detail}</small><img src='{uri(rec['path'])}'></figure>")
        out.append("</div></section>")
    note=("Structured exports physically remove FFN neurons and attention heads; .pth files contain complete smaller UNet objects." if structured else "Unstructured exports retain dense tensor shapes with zero weights; dense CUDA kernels do not automatically produce sparse speedups.")
    out.append(f"<section class='card'><p>{note}</p></section>")
    out.append("<section class='card'><pre>"+html.escape(json.dumps(data,indent=2))+"</pre></section></body></html>")
    path.write_text("".join(out),encoding="utf-8")


def main():
    a=parse_args(); root=Path(a.compat_root).resolve(); prepare(root); utils,source=import_inference()
    data=json.loads(Path(a.input_json).read_text()); device="cuda" if torch.cuda.is_available() else "cpu"
    model=utils.load(name=str(Path(a.checkpoint).resolve()),device=device,med_config=str(Path(a.med_config).resolve()))
    variants=list(data["variants"].keys())
    for i,case in enumerate(data["cases"],1):
        print(f"Case {i}/{len(data['cases'])}: {case['prompt']}")
        for v in variants:
            with torch.inference_mode(): value=score_value(model.score(case["prompt"],case["images"][v]["path"]))
            case["images"][v]["imagereward"]=value; print(f"  {v}: {value:+.6f}")
    summary={}
    dense=[float(c["images"]["dense"]["imagereward"]) for c in data["cases"]]
    for v in variants:
        scores=[float(c["images"][v]["imagereward"]) for c in data["cases"]]
        deltas=[x-y for x,y in zip(scores,dense)]
        summary[v]={"cases":len(scores),"mean_imagereward":float(np.mean(scores)),"mean_delta_vs_dense":float(np.mean(deltas)),"median_delta_vs_dense":float(np.median(deltas)),"minimum_delta_vs_dense":float(np.min(deltas)),"maximum_delta_vs_dense":float(np.max(deltas)),"wins_vs_dense":int(sum(x>0 for x in deltas)),"losses_vs_dense":int(sum(x<0 for x in deltas))}
    data["imagereward_summary"]=summary
    data["imagereward_environment"]={"transformers":__import__("transformers").__version__,"image_reward_source":str(source),"refl_imported":"ImageReward.ReFL" in sys.modules,"datasets_imported":"datasets" in sys.modules,"diffusers_imported":"diffusers" in sys.modules}
    Path(a.output_json).write_text(json.dumps(data,indent=2)); build(data,Path(a.output_html))
    print(f"Scored JSON: {a.output_json}\nScored HTML: {a.output_html}")


if __name__=="__main__": main()

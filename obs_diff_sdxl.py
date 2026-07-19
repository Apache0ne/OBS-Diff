#!/usr/bin/env python3
"""Timestep-aware OBS-Diff pruning for SDXL single-file or Diffusers models.

Calibrates the dense UNet once, creates independent 20/30/40/50% unstructured
variants from the same full Hessians, exports loadable Diffusers UNets, and
builds a dense-vs-pruned comparison report. Tensor shapes remain dense; sparse
kernels are required for actual sparse acceleration.
"""
from __future__ import annotations

import argparse, base64, gc, html, json, math, random, shutil, time
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file, save_file
from diffusers import DPMSolverSinglestepScheduler, StableDiffusionXLPipeline

TARGETS = (
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
)
CAL_PROMPTS = [
    "a studio photograph of a red fox sitting in fresh snow, detailed fur",
    "an old stone lighthouse above a stormy ocean at sunset",
    "a modern glass house in a pine forest, architectural photography",
    "a bowl of fruit on a wooden table, soft window light",
    "a vintage blue automobile parked on a city street",
    "a close portrait of an astronaut wearing a reflective helmet",
    "a small robot reading a book in a quiet library",
    "a mountain village beneath the northern lights",
]
TEST_PROMPTS = [
    "AN ADULT BEAR IS STANDING IN THE FIELD",
    "an odd looking toilet is against a wall",
    "A bathroom scene is shown with a tub and counter.",
    "a large plane is flying in the sky",
]
STEP = {"value": 0}


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", default="/content/obs_diff_sdxl_results")
    p.add_argument("--ratios", default="0.20,0.30,0.40,0.50")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--calibration-prompts", type=int, default=8)
    p.add_argument("--calibration-size", type=int, default=512)
    p.add_argument("--compare-size", type=int, default=1024)
    p.add_argument("--coco-captions", default=None)
    p.add_argument("--package-hessian-gib", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--column-block", type=int, default=128)
    p.add_argument("--save-unets", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-tf32", action="store_true")
    return p.parse_args()


def ratios(raw: str) -> List[float]:
    values = sorted(set(float(x.strip()) for x in raw.split(",") if x.strip()))
    if not values or any(x <= 0 or x >= 1 for x in values):
        raise ValueError("Ratios must satisfy 0 < ratio < 1")
    return values


def label(r: float) -> str:
    return f"{int(round(r * 100)):02d}"


def scheduler(pipe):
    cfg = dict(pipe.scheduler.config)
    cfg.update(dict(
        solver_order=2, algorithm_type="sde-dpmsolver++", solver_type="midpoint",
        lower_order_final=True, thresholding=False, use_karras_sigmas=False,
        use_exponential_sigmas=False, use_beta_sigmas=False,
        final_sigmas_type="zero", steps_offset=0,
    ))
    try:
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(cfg)
    except TypeError:
        for key in ("use_exponential_sigmas", "use_beta_sigmas", "final_sigmas_type"):
            cfg.pop(key, None)
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(cfg)


def load_pipe(a):
    dtype = torch.float16 if a.dtype == "float16" else torch.bfloat16
    path = Path(a.model)
    kw = dict(torch_dtype=dtype, local_files_only=a.local_files_only)
    if path.is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(path), use_safetensors=path.suffix.lower() == ".safetensors", **kw
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(str(path), **kw)
    scheduler(pipe)
    pipe.enable_vae_tiling(); pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda"); pipe.unet.eval()
    return pipe


def get_prompts(a) -> Tuple[List[str], List[str]]:
    cal: List[str] = []
    if a.coco_captions and Path(a.coco_captions).is_file():
        data = json.loads(Path(a.coco_captions).read_text())
        cal = [str(x.get("caption", "")).strip() for x in data.get("annotations", [])]
        cal = [x for x in cal if x]
        random.Random(a.seed).shuffle(cal)
    cal = (cal + CAL_PROMPTS)[:a.calibration_prompts]
    return cal, TEST_PROMPTS


def target_modules(unet) -> "OrderedDict[str, nn.Linear]":
    out = OrderedDict()
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and ".transformer_blocks." in f".{name}" and any(name.endswith(s) for s in TARGETS):
            out[name] = module
    if not out:
        raise RuntimeError("No SDXL BasicTransformerBlock Linear targets found")
    return out


def packages(modules: Mapping[str, nn.Linear], budget: int) -> List[List[str]]:
    out, current, used = [], [], 0
    for name, module in modules.items():
        cols = module.weight.shape[1]
        cost = cols * cols * 4
        if current and used + cost > budget:
            out.append(current); current, used = [], 0
        current.append(name); used += cost
    if current: out.append(current)
    return out


def step_callback(pipe, step, timestep, kwargs):
    STEP["value"] = int(step) + 1
    return kwargs


def timestep_weights(steps: int) -> np.ndarray:
    if steps <= 1: return np.ones(steps)
    x = np.arange(steps)
    return (0.8 + 0.4 / np.log(steps) * np.log1p(x))[::-1].copy()


class Hessian:
    def __init__(self, cols: int, max_tokens: int, device):
        self.cols, self.max_tokens = cols, max_tokens
        self.H = torch.zeros((cols, cols), dtype=torch.float32, device=device)
        self.weight = 0.0; self.calls = 0; self.tokens = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor, weight: float):
        x = x.reshape(-1, x.shape[-1])
        if x.shape[1] != self.cols: raise RuntimeError("Hook width mismatch")
        if self.max_tokens and x.shape[0] > self.max_tokens:
            idx = torch.linspace(0, x.shape[0]-1, self.max_tokens, device=x.device).round().long()
            x = x.index_select(0, idx)
        x = x.float(); new = weight * x.shape[0]; total = self.weight + new
        if self.weight: self.H.mul_(self.weight / total)
        x.mul_(math.sqrt(2.0 * weight / total)); self.H.addmm_(x.t(), x)
        self.weight = total; self.calls += 1; self.tokens += x.shape[0]


def calibrate(a, pipe, names, modules, prompts, weights):
    stats: Dict[str, Hessian] = {}; hooks = []
    for name in names:
        module = modules[name]; h = Hessian(module.weight.shape[1], a.max_tokens, module.weight.device)
        stats[name] = h
        def hook(mod, inp, out, h=h):
            h.add(inp[0].detach(), float(weights[min(STEP["value"], len(weights)-1)]))
        hooks.append(module.register_forward_hook(hook))
    try:
        for i, prompt in enumerate(prompts):
            STEP["value"] = 0
            pipe(
                prompt=prompt, height=a.calibration_size, width=a.calibration_size,
                num_inference_steps=a.steps, guidance_scale=a.guidance_scale,
                generator=torch.Generator("cuda").manual_seed(a.seed+i),
                output_type="latent", callback_on_step_end=step_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
    finally:
        for hook in hooks: hook.remove()
    out = {}
    for name, h in stats.items():
        if not h.weight: raise RuntimeError(f"No samples reached {name}")
        print(f"  {name}: H={h.cols} calls={h.calls} tokens={h.tokens}")
        out[name] = h.H.cpu(); h.H = None
    torch.cuda.empty_cache(); return out


def hessian_inverse(Hcpu: torch.Tensor, damp: float):
    H = Hcpu.cuda(non_blocking=True).float(); dead = torch.diag(H) == 0
    H[dead, dead] = 1; diag = torch.arange(H.shape[0], device=H.device)
    base = max(float(torch.mean(torch.diag(H))) * damp, 1e-8)
    error = None
    for mult in (1, 10, 100, 1000):
        trial = H.clone(); trial[diag, diag] += base * mult
        try:
            chol = torch.linalg.cholesky(trial)
            return torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True), dead.cpu()
        except Exception as exc:
            error = exc; del trial; torch.cuda.empty_cache()
    raise RuntimeError("Hessian Cholesky failed") from error


@torch.no_grad()
def obs_weight(dense: torch.Tensor, Hinv: torch.Tensor, dead: torch.Tensor, ratio: float, block: int):
    W = dense.cuda(non_blocking=True).float(); dead = dead.cuda(non_blocking=True)
    if dead.any(): W[:, dead] = 0
    cols = W.shape[1]
    for s in range(0, cols, block):
        e = min(s+block, cols); n = e-s
        W1 = W[:, s:e].clone(); Q = torch.zeros_like(W1); E = torch.zeros_like(W1); Hi = Hinv[s:e, s:e]
        score = W1.square() / torch.diag(Hi).reshape(1,-1).square().clamp_min(1e-20)
        k = min(max(int(score.numel()*ratio), 0), score.numel()-1)
        mask = score <= torch.kthvalue(score.flatten(), k+1).values
        for j in range(n):
            w = W1[:,j]; q = w.clone(); q[mask[:,j]] = 0; Q[:,j] = q
            err = (w-q) / Hi[j,j]
            if j+1 < n: W1[:,j+1:] -= err[:,None] @ Hi[j,j+1:][None,:]
            E[:,j] = err
        W[:,s:e] = Q
        if e < cols: W[:,e:] -= E @ Hinv[s:e,e:]
    return W.to(dtype=dense.dtype, device="cpu").contiguous()


def make_shards(a, root: Path, package_id: int, names, modules, stats, rs):
    tensors = {r:{} for r in rs}
    for i, name in enumerate(names, 1):
        dense = modules[name].weight.detach().cpu().contiguous()
        print(f"  OBS {i}/{len(names)} {name} {tuple(dense.shape)}")
        Hinv, dead = hessian_inverse(stats[name], a.percdamp)
        for r in rs:
            start = time.perf_counter(); w = obs_weight(dense, Hinv, dead, r, a.column_block)
            tensors[r][name+".weight"] = w
            print(f"    {r:.0%}: {(w==0).sum().item()/w.numel():.6f} in {time.perf_counter()-start:.2f}s")
        del dense, Hinv, dead; torch.cuda.empty_cache()
    for r in rs:
        folder = root/"shards"/f"sparsity_{label(r)}"; folder.mkdir(parents=True, exist_ok=True)
        save_file(tensors[r], str(folder/f"package_{package_id:03d}.safetensors"), metadata={"method":"OBS-Diff","ratio":str(r)})


def module_by_name(root, name):
    obj = root
    for part in name.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def apply_shards(unet, root: Path, ratio: float):
    count = 0
    for shard in sorted((root/"shards"/f"sparsity_{label(ratio)}").glob("*.safetensors")):
        for key, tensor in load_file(str(shard), device="cpu").items():
            name = key[:-7]; module = module_by_name(unet, name)
            module.weight.data.copy_(tensor.to(module.weight.device, module.weight.dtype)); count += 1
    if not count: raise RuntimeError(f"No shards for ratio {ratio}")
    print(f"Applied {count} weights")


def sparse_stats(unet, modules):
    tt = tz = 0
    for module in modules.values():
        tt += module.weight.numel(); tz += int((module.weight==0).sum())
    ut = uz = 0
    for p in unet.parameters():
        ut += p.numel(); uz += int((p==0).sum())
    return {"target_total":tt,"target_zeros":tz,"target_sparsity":tz/tt,"unet_total":ut,"unet_zeros":uz,"overall_unet_zero_fraction":uz/ut}


def generate(a, pipe, prompts, variant, root):
    folder = root/"images"/variant; folder.mkdir(parents=True, exist_ok=True)
    rows=[]; times=[]
    for i,prompt in enumerate(prompts):
        seed=a.seed+i
        if i==0:
            pipe(prompt=prompt,height=a.compare_size,width=a.compare_size,num_inference_steps=a.steps,guidance_scale=a.guidance_scale,generator=torch.Generator("cuda").manual_seed(seed),output_type="latent")
        torch.cuda.synchronize(); start=time.perf_counter()
        image=pipe(prompt=prompt,height=a.compare_size,width=a.compare_size,num_inference_steps=a.steps,guidance_scale=a.guidance_scale,generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        path=folder/f"case_{i:02d}.png"; image.save(path); times.append(elapsed)
        rows.append({"index":i,"prompt":prompt,"seed":seed,"path":str(path),"generation_seconds":elapsed})
        print(f"  {variant} {i}: {elapsed:.3f}s")
    return rows, float(np.mean(times))


def metrics(ref, candidate):
    a=np.asarray(Image.open(ref).convert("RGB"),np.float32); b=np.asarray(Image.open(candidate).convert("RGB"),np.float32)
    d=b-a; mse=float(np.mean(d*d)); rmse=math.sqrt(mse)
    return {"psnr_db":float("inf") if mse==0 else 20*math.log10(255/rmse),"mae_0_1":float(np.mean(np.abs(d)))/255,"rmse_0_1":rmse/255,"pixels_gt_16_pct":float(np.mean(np.max(np.abs(d),axis=2)>16)*100)}


def uri(path):
    buf=BytesIO(); Image.open(path).convert("RGB").save(buf,"JPEG",quality=90,optimize=True)
    return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()


def report(data, path: Path):
    rs=data["config"]["ratios"]; variants=["dense"]+[f"sparsity_{label(r)}" for r in rs]
    css="body{font-family:Arial;background:#f3f4f6;margin:20px}.card{background:white;padding:15px;margin:12px 0;border:1px solid #ccc}.grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:8px;overflow-x:auto}img{width:100%}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:6px}pre{white-space:pre-wrap;background:#111;color:#eee;padding:10px}"
    out=[f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body><h1>OBS-Diff SDXL comparison</h1>"]
    out.append("<section class='card'><p>Full-Hessian, timestep-aware OBS calibration was run once. Each ratio was produced independently from dense package weights. Dense tensor shapes are retained.</p></section>")
    out.append("<section class='card'><table><tr><th>Variant</th><th>Target sparsity</th><th>Overall UNet zeros</th><th>Mean seconds</th><th>UNet</th></tr>")
    for v in variants:
        s=data["variants"][v]; out.append(f"<tr><td>{v}</td><td>{100*s.get('target_sparsity',0):.4f}%</td><td>{100*s.get('overall_unet_zero_fraction',0):.4f}%</td><td>{s['mean_generation_seconds']:.4f}</td><td>{html.escape(str(s.get('unet_dir')))}</td></tr>")
    out.append("</table></section>")
    for case in data["cases"]:
        out.append(f"<section class='card'><h2>{html.escape(case['prompt'])}</h2><div class='grid'>")
        for v in variants:
            rec=case["images"][v]; detail=f"{rec['generation_seconds']:.3f}s"
            if v!="dense":
                m=case["metrics"][v]; detail+=f" | PSNR {m['psnr_db']:.2f} | MAE {m['mae_0_1']:.4f}"
            out.append(f"<figure><b>{v}</b><br><small>{detail}</small><img src='{uri(rec['path'])}'></figure>")
        out.append("</div></section>")
    out.append("<section class='card'><pre>"+html.escape(json.dumps(data,indent=2))+"</pre></section></body></html>")
    path.write_text("".join(out),encoding="utf-8")


def main():
    a=args_parser()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.matmul.allow_tf32=a.allow_tf32; torch.backends.cudnn.allow_tf32=a.allow_tf32
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rs=ratios(a.ratios); root=Path(a.output_dir).resolve()
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    cal,test=get_prompts(a); pipe=load_pipe(a); modules=target_modules(pipe.unet)
    total=sum(p.numel() for p in pipe.unet.parameters()); target=sum(m.weight.numel() for m in modules.values())
    model_stats={"unet_parameters":total,"target_weight_parameters":target,"target_coverage_of_unet":target/total,"target_modules":len(modules)}
    packs=packages(modules,int(a.package_hessian_gib*1024**3)); weights=timestep_weights(a.steps)
    print(json.dumps(model_stats,indent=2)); print(f"Packages: {len(packs)} timestep weights: {weights.tolist()}")
    started=time.perf_counter()
    for i,names in enumerate(packs):
        print("\n"+"="*80+f"\nPACKAGE {i+1}/{len(packs)} modules={len(names)}\n"+"="*80)
        stats=calibrate(a,pipe,names,modules,cal,weights); make_shards(a,root,i,names,modules,stats,rs)
        del stats; gc.collect(); torch.cuda.empty_cache()
    prune_seconds=time.perf_counter()-started
    dense,dense_time=generate(a,pipe,test,"dense",root)
    cases=[{"index":r["index"],"prompt":r["prompt"],"seed":r["seed"],"images":{"dense":r},"metrics":{}} for r in dense]
    variants={"dense":{"target_sparsity":0.0,"overall_unet_zero_fraction":sparse_stats(pipe.unet,modules)["overall_unet_zero_fraction"],"mean_generation_seconds":dense_time,"unet_dir":None}}
    for r in rs:
        v=f"sparsity_{label(r)}"; print("\n"+"="*80+f"\nAPPLY {r:.0%}\n"+"="*80)
        apply_shards(pipe.unet,root,r); s=sparse_stats(pipe.unet,modules); unet_dir=None
        if a.save_unets:
            dest=root/"unets"/v; dest.parent.mkdir(parents=True,exist_ok=True)
            pipe.unet.save_pretrained(str(dest),safe_serialization=True,max_shard_size="5GB"); unet_dir=str(dest)
            (dest/"obs_diff_manifest.json").write_text(json.dumps({"base_model":str(Path(a.model).resolve()),"method":"OBS-Diff","ratio":r,**s},indent=2))
        rows,mean_time=generate(a,pipe,test,v,root)
        for case,row in zip(cases,rows):
            case["images"][v]=row; case["metrics"][v]=metrics(case["images"]["dense"]["path"],row["path"])
        variants[v]={**s,"mean_generation_seconds":mean_time,"unet_dir":unet_dir}
    data={"config":{"model":str(Path(a.model).resolve()),"ratios":rs,"steps":a.steps,"guidance_scale":a.guidance_scale,"seed":a.seed,"calibration_prompts":cal,"compare_prompts":test,"timestep_weights":weights.tolist(),"pruning_seconds":prune_seconds,"target_suffixes":list(TARGETS)},"model_stats":model_stats,"variants":variants,"cases":cases}
    jp=root/"obs_sdxl_compare.json"; hp=root/"obs_sdxl_compare.html"
    jp.write_text(json.dumps(data,indent=2)); report(data,hp)
    print(f"\nCOMPLETE\nJSON: {jp}\nHTML: {hp}\nUNets: {root/'unets'}")


if __name__ == "__main__": main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGPS – Bullet‑proof Llama pretraining_tp patch
"""

# ─── 기본 import ───
import argparse, random, json, time, re, sys, importlib.util
from pathlib import Path
from typing import List

REQUIRED_PKGS = ["numpy", "torch", "evaluate", "detoxify",
                 "datasets", "transformers", "bitsandbytes"]

def check_requirements():
    missing = []
    for name in REQUIRED_PKGS:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    if missing:
        sys.stderr.write(
            "Missing required packages: {}\n".format(", ".join(missing)))
        sys.stderr.write("Please install dependencies listed in requirements.txt\n")
        return False
    return True

if not check_requirements():
    sys.exit(1)

import numpy as np
from tqdm.auto import tqdm
from loguru import logger
import torch

# ─── transformers 글로벌 패치 ───
from transformers import AutoTokenizer, AutoModel, AutoConfig
import bitsandbytes as bnb  # 이미 설치돼 있다면, 없으면 4‑bit 비활성

# 1) 재귀 함수: 어떤 컨테이너 안에서도 Config 찾아 세팅
def _set_tp(obj):
    from transformers import PretrainedConfig
    if isinstance(obj, PretrainedConfig):
        if getattr(obj, "pretraining_tp", None) is None:
            obj.pretraining_tp = 1
        if getattr(obj, "parallelization_style", None) is None:
            obj.parallelization_style = []
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _set_tp(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            _set_tp(v)

# 2) 원본 보존 후 패치
_orig_ac = AutoConfig.from_pretrained
def _patched_ac_from_pretrained(*args, **kw):
    ret = _orig_ac(*args, **kw)
    _set_tp(ret)
    return ret
AutoConfig.from_pretrained = _patched_ac_from_pretrained
# ──────────────────────────────────────

# ─── Diffusion‑LM 로더 ───
from transformers import BitsAndBytesConfig
def load_dlm(model_id="diffusionfamily/diffullama", four_bit=True):
    if four_bit:
        try:
            quant = BitsAndBytesConfig(load_in_4bit=True,
                                       bnb_4bit_compute_dtype=torch.bfloat16)
        except Exception:
            logger.warning("bitsandbytes 문제 → 4‑bit 끔, bf16 로드")
            quant, four_bit = None, False
    else:
        quant = None

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if not four_bit else None,
    )
    model.eval()
    return tok, model

# ─────────── saliency(∇x log p) 계산 ───────────
@torch.inference_mode()
def token_saliency(model, tok, prompt: str, num_steps: int = 12) -> List[float]:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if hasattr(model, "compute_scores"):
        s = model.compute_scores(
            inputs["input_ids"], num_steps=num_steps, return_all_scores=True
        )
        return s.norm(dim=-1).mean(dim=0).cpu().tolist()

    inputs["input_ids"].requires_grad_(True)
    out = model(**inputs, diffusion_steps=num_steps)
    (out.loss if hasattr(out, "loss") else out[0].sum()).backward()
    g = inputs["input_ids"].grad.float()
    return g.abs().sum(dim=-1).squeeze().tolist()

# ─────────────── 평가 메트릭 ───────────────
import evaluate
from detoxify import Detoxify
bleu_metric = evaluate.load("sacrebleu")
tox_model   = Detoxify("original",
                       device="cuda" if torch.cuda.is_available() else "cpu")

def calc_bleu(h, r): return bleu_metric.compute(predictions=h,
                                                references=[[x] for x in r])["score"]
def toxic(x): return float(np.mean(tox_model.predict(x)["toxicity"]))

# ─────────────── 데이터 로더 ───────────────
from datasets import load_dataset, Dataset
def load_parallel(name, subset, split, src, tgt, n, token):
    ds: Dataset = load_dataset(name, subset, split=split,
                               token=token)  # token=None OK
    if 0 < n < len(ds):
        ds = ds.shuffle(seed=0).select(range(n))

    if "translation" in ds.features:
        srcs = [ex["translation"][src] for ex in ds]
        refs = [ex["translation"][tgt] for ex in ds]
    elif {"source", "target"} <= set(ds.column_names):
        srcs, refs = ds["source"], ds["target"]
    else:
        raise ValueError("지원되지 않는 포맷")
    return srcs, refs

# ─────────────── GA 구성 요소 ───────────────
class Individual:
    def __init__(self, prompt): self.prompt, self.score, self.low = prompt, None, None
def mutate(ind, rate=0.3):
    w = ind.prompt.split()
    for idx in ind.low:
        if idx < len(w) and random.random() < rate:
            if random.random() < .5 and len(w) > 3: w[idx] = ""
            else: w[idx] = random.choice(
                ["Let's think step‑by‑step.", "Firstly,", "In summary,"])
    return " ".join(filter(None, w))
def crossover(a, b):
    wa, wb = a.prompt.split(), b.prompt.split()
    cut = random.randint(1, min(len(wa), len(wb))-1)
    return " ".join(wa[:cut] + wb[cut:])

# ─────────────── 적합도 ───────────────
_cache = {}
def fitness(ind, model, tok, srcs, refs, args):
    if ind.prompt in _cache:
        ind.score, ind.low = _cache[ind.prompt]; return ind.score

    sal = token_saliency(model, tok, ind.prompt, args.steps)
    thr = np.percentile(sal, 30); ind.low = [i for i,s in enumerate(sal) if s < thr]

    hyps = []
    for s in srcs:
        ids = tok(f"{ind.prompt}\n{s}", return_tensors="pt").to(model.device)
        out = model.generate_diffusion(ids.input_ids, num_inference_steps=args.steps,
                                       guidance_scale=1.1, max_new_tokens=128)
        hyps.append(tok.decode(out[0], skip_special_tokens=True))

    score = calc_bleu(hyps, refs) - args.beta*toxic(hyps) - args.gamma*len(ind.prompt.split())
    ind.score = score; _cache[ind.prompt] = (score, ind.low)
    return score

# ─────────────── GA 메인 루프 ───────────────
def run_ga(a):
    srcs, refs = load_parallel(a.dataset_name, a.dataset_subset, a.split,
                               a.src_lang, a.tgt_lang, a.samples, a.hf_token)
    logger.info(f"{len(srcs)} pairs loaded from {a.dataset_name}/{a.dataset_subset}")

    tok, model = load_dlm(a.model_id, four_bit=not a.fp16)

    seed = (f"You are a helpful translation assistant. "
            f"Translate the following {a.src_lang.upper()} sentence into {a.tgt_lang.upper()}.")
    pop = [Individual(seed) for _ in range(a.pop_size)]

    run_dir = Path("runs")/time.strftime("%Y%m%d-%H%M%S"); run_dir.mkdir(parents=True)

    for g in range(a.gens):
        logger.info(f"=== Gen {g} ===")
        for ind in tqdm(pop, desc="eval"): fitness(ind, model, tok, srcs, refs, a)
        pop.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"best composite: {pop[0].score:.2f}")

        json.dump([vars(i) for i in pop], open(run_dir/f"gen{g}.json","w"), indent=2)
        surv = pop[:len(pop)//2]; children=[]
        while len(surv)+len(children) < a.pop_size:
            child = mutate(random.choice(surv)) if random.random()<.5 \
                    else crossover(*random.sample(surv,2))
            children.append(Individual(child))
        pop = surv+children

    (run_dir/"best_prompt.txt").write_text(pop[0].prompt)
    logger.success(f"Done → {run_dir/'best_prompt.txt'}")

# ─────────────── CLI ───────────────
def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="iwslt2017")
    p.add_argument("--dataset_subset", default="iwslt2017-en-fr")
    p.add_argument("--split", default="test[:200]")
    p.add_argument("--src_lang", default="en"); p.add_argument("--tgt_lang", default="fr")
    p.add_argument("--samples", type=int, default=200); p.add_argument("--hf_token")
    p.add_argument("--model_id", default="diffusionfamily/diffullama"); p.add_argument("--fp16",action="store_true")
    p.add_argument("--steps", type=int, default=12); p.add_argument("--pop_size", type=int, default=12)
    p.add_argument("--gens", type=int, default=6); p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.02)
    return p.parse_args()

if __name__ == "__main__":
    run_ga(cli())

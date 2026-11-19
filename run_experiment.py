#!/usr/bin/env python
"""Rank-1 weight edit vs. activation steering vs. prompting for Swedish language fidelity.

QUESTION
    Can a tiny, targeted weight edit (one direction removed from the MLP output weights,
    as in the "Weight-Space Programming" paper's Qwen experiment) make a small LM answer
    Swedish prompts in Swedish, without damaging its language modelling?

PIPELINE
    1. Measure the unchanged model on DEV prompts (baseline).
    2. Grid-search edit/steering settings on DEV only (layer range, strength).
    3. Freeze the chosen settings (selection rule: highest Swedish rate subject to a
       perplexity limit), then evaluate ONCE on TEST prompts, together with controls:
       unchanged model, prompt baseline, random-direction edits, additive steering.
    4. Save numbers (results/results.json) and example generations (results/examples.md).

Run:  python run_experiment.py --config config.yaml
"""
import argparse
import contextlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "results"

# Candidate languages for the language detector. Restricting the set makes detection
# faster and more accurate; Danish/Bokmal are included because they are the most likely
# confusions with Swedish, Chinese because Qwen models sometimes drift into it.
LANGS = [Language.SWEDISH, Language.ENGLISH, Language.DANISH, Language.BOKMAL,
         Language.GERMAN, Language.FINNISH, Language.CHINESE, Language.FRENCH,
         Language.SPANISH, Language.RUSSIAN]


# ----------------------------------------------------------------------------- utilities
def read_lines(path):
    """Read a UTF-8 text file into a list of non-empty, stripped lines."""
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n.
    Preferred over the normal approximation because n is small (30 prompts)."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def lang_stats(detector, outs):
    """Detect the language of each generated answer; report the Swedish share,
    its confidence interval, and the full language counts (to see Danish/Norwegian
    confusions or English/Chinese drift)."""
    labels = []
    for o in outs:
        lang = detector.detect_language_of(o) if o.strip() else None
        labels.append(lang.name if lang else "NONE")
    k = sum(l == "SWEDISH" for l in labels)
    return {"sv_rate": k / len(labels), "ci95": wilson(k, len(labels)),
            "lang_counts": dict(Counter(labels))}


# ----------------------------------------------------------------------------- model ops
@torch.no_grad()
def generate(model, tok, prompts, system, max_new_tokens, batch_size):
    """Greedy (deterministic) chat generation for a list of prompts.
    `system` is an optional system prompt (used only for the prompting baseline)."""
    tok.padding_side = "left"  # decoder-only models must be left-padded for batched generation
    outs = []
    for i in range(0, len(prompts), batch_size):
        texts = []
        for p in prompts[i:i + batch_size]:
            msgs = ([{"role": "system", "content": system}] if system else [])
            msgs.append({"role": "user", "content": p})
            # Use the model's own chat template so the prompt format is correct.
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        # add_special_tokens=False: the template already contains any special tokens.
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        # Keep only newly generated tokens (strip the prompt part).
        outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


@torch.no_grad()
def perplexity(model, tok, texts):
    """Token-weighted perplexity over a list of passages. Used as a regression check:
    if an edit raises perplexity a lot, it damaged general language modelling."""
    nll, n = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt").to(model.device)
        loss = model(**enc, labels=enc.input_ids).loss.item()  # mean NLL per predicted token
        k = enc.input_ids.shape[1] - 1                          # number of predicted tokens
        nll += loss * k
        n += k
    return math.exp(nll / n)


@torch.no_grad()
def mean_hidden(model, tok, texts, layer):
    """Mean residual-stream vector after decoder layer `layer`, averaged over tokens and
    texts. The first token is skipped because it often has an abnormally large norm
    (attention-sink effect) that would dominate the mean. Also returns the mean token norm,
    which is later used to scale the steering strength."""
    vecs, norms = [], []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(model.device)
        # hidden_states[0] is the embedding output, so layer i's output is index i + 1.
        hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1][0, 1:].float()
        vecs.append(hs.mean(0))
        norms.append(hs.norm(dim=-1).mean())
    return torch.stack(vecs).mean(0), torch.stack(norms).mean().item()


@contextlib.contextmanager
def weight_edit(model, layers, d, lam):
    """Rank-1 edit of MLP output weights: W' = (I - lam * d d^T) W.

    W = mlp.down_proj.weight has shape (hidden, intermediate) and maps MLP activations to
    what the MLP writes into the residual stream. The edit removes the component of that
    output along the unit vector d (fully if lam = 1, partially if lam < 1).
    The original weights are restored on exit, so every configuration starts from the
    same unedited model."""
    saved = {}
    d = d.to(model.device, torch.float32)
    try:
        for i in layers:
            W = model.model.layers[i].mlp.down_proj.weight
            saved[i] = W.data.clone()
            Wf = W.data.float()
            # outer(d, d @ W) = d d^T W, the part of W that writes along direction d.
            W.data.copy_((Wf - lam * torch.outer(d, d @ Wf)).to(W.dtype))
        yield
    finally:
        for i, w in saved.items():
            model.model.layers[i].mlp.down_proj.weight.data.copy_(w)


@contextlib.contextmanager
def steering(model, layer, vec):
    """Conventional alternative: additive activation steering at inference time.
    Adds `vec` to the residual stream after decoder layer `layer` via a forward hook;
    the hook is removed on exit. No weights are changed."""
    vec = vec.to(model.device)

    def hook(_mod, _inp, out):
        # Depending on the transformers version a decoder layer returns a tuple or a tensor.
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
        return out + vec.to(out.dtype)

    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ROOT / ap.parse_args().config))
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ---- load model (float32: small model, avoids half-precision issues on a T4) -----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], revision=cfg["revision"], torch_dtype=torch.float32).to(device).eval()
    # Record the exact model version so the run is reproducible (copy it into config.yaml).
    commit = getattr(model.config, "_commit_hash", None)
    print(f"model={cfg['model_name']} commit={commit} device={device}")

    detector = LanguageDetectorBuilder.from_languages(*LANGS).build()

    # ---- data -----------------------------------------------------------------------------
    dev = read_lines(DATA / "dev_prompts.txt")    # used for choosing settings
    test = read_lines(DATA / "test_prompts.txt")  # used only for the final evaluation
    # Parallel Swedish/English sentences: used ONLY to estimate the language direction.
    pairs = [l.split("\t") for l in read_lines(DATA / "direction_pairs.tsv")]
    sv_txt, en_txt = [p[0] for p in pairs], [p[1] for p in pairs]
    ppl_sv, ppl_en = read_lines(DATA / "ppl_sv.txt"), read_lines(DATA / "ppl_en.txt")
    h_sv = len(ppl_sv) // 2
    h_en = len(ppl_en) // 2
    ppl_dev = (ppl_sv[:h_sv], ppl_en[:h_en])      # first half: used for selection
    ppl_tst = (ppl_sv[h_sv:], ppl_en[h_en:])      # second half: final report only

    n_layers = len(model.model.layers)
    hidden = model.config.hidden_size

    def layers_of(fr):
        """Convert a (start, end) fraction of model depth into a list of layer indices."""
        lo = int(round(fr[0] * n_layers))
        hi = max(lo + 1, int(round(fr[1] * n_layers)))
        return list(range(lo, min(hi, n_layers)))

    dir_cache = {}

    def direction(layer):
        """Unit 'Swedish minus English' direction at `layer` (difference of mean activations
        over the parallel sentences), plus the mean token norm. Cached per layer."""
        if layer not in dir_cache:
            m_sv, n1 = mean_hidden(model, tok, sv_txt, layer)
            m_en, n2 = mean_hidden(model, tok, en_txt, layer)
            d = m_sv - m_en
            dir_cache[layer] = (d / d.norm(), (n1 + n2) / 2)
        return dir_cache[layer]

    def measure(make_ctx, prompts, ppl_texts, system=None):
        """Evaluate the model under an intervention (`make_ctx` returns a context manager):
        generate answers for `prompts`, detect their language, and compute Swedish and
        English perplexity under the SAME intervention."""
        with make_ctx():
            outs = generate(model, tok, prompts, system, cfg["max_new_tokens"], cfg["batch_size"])
            p_sv = perplexity(model, tok, ppl_texts[0])
            p_en = perplexity(model, tok, ppl_texts[1])
        r = lang_stats(detector, outs)
        r.update(ppl_sv=p_sv, ppl_en=p_en, mean_chars=float(np.mean([len(o) for o in outs])))
        return r, outs

    none = contextlib.nullcontext  # "no intervention"

    # ---- sanity check: after a full edit the direction must be (numerically) removed -------
    l0 = layers_of(cfg["layer_ranges"][0])[0]
    d0 = direction(l0)[0]
    with weight_edit(model, [l0], d0, 1.0):
        resid = (d0 @ model.model.layers[l0].mlp.down_proj.weight.data.float()).norm().item()
    print(f"selfcheck: ||d^T W'|| after full edit = {resid:.2e} (should be ~0)")

    # ---- DEV: baseline + grid search -------------------------------------------------------
    base_dev, _ = measure(none, dev, ppl_dev)
    print("dev baseline:", {k: base_dev[k] for k in ("sv_rate", "ppl_sv", "ppl_en")})
    # Regression guard: an edit is admissible only if Swedish perplexity rises <= 10%.
    limit = base_dev["ppl_sv"] * (1 + cfg["max_ppl_increase"])
    grid = []
    for fr in cfg["layer_ranges"]:
        layers = layers_of(fr)
        center = layers[len(layers) // 2]   # direction is estimated at the middle layer
        d, nrm = direction(center)
        for lam in cfg["lambdas"]:
            # Weight edit on all layers in the range, same direction d in each.
            r, _ = measure(lambda: weight_edit(model, layers, d, lam), dev, ppl_dev)
            grid.append(dict(method="edit", layers=layers, center=center, lam=lam, **r))
            print("dev edit", layers[0], layers[-1], lam, round(r["sv_rate"], 3), round(r["ppl_sv"], 2))
        for al in cfg["alphas"]:
            # Steering at one layer; strength is a fraction of the typical token norm.
            vec = al * nrm * d
            r, _ = measure(lambda: steering(model, center, vec), dev, ppl_dev)
            grid.append(dict(method="steer", layers=[center], center=center, alpha=al, **r))
            print("dev steer", center, al, round(r["sv_rate"], 3), round(r["ppl_sv"], 2))

    def pick(method):
        """Best admissible dev configuration: highest Swedish rate, ties broken by lower
        perplexity. Returns None if no configuration respects the perplexity limit."""
        ok = [g for g in grid if g["method"] == method and g["ppl_sv"] <= limit]
        return max(ok, key=lambda g: (g["sv_rate"], -g["ppl_sv"])) if ok else None

    best_edit, best_steer = pick("edit"), pick("steer")
    print("selected edit:", best_edit and (best_edit["layers"], best_edit["lam"]))
    print("selected steer:", best_steer and (best_steer["center"], best_steer["alpha"]))

    # ---- TEST (configuration frozen; test prompts never used above) -------------------------
    results = {"commit": commit, "config": cfg, "n_test_prompts": len(test),
               "dev_baseline": base_dev, "dev_grid": grid,
               "selected": {"edit": best_edit, "steer": best_steer}, "test": {}}
    examples = {}

    # Control 1: unchanged model.
    r, examples["baseline"] = measure(none, test, ppl_tst)
    results["test"]["baseline"] = r
    # Control 2: simple prompting alternative ("always answer in Swedish").
    r, examples["prompt_sv"] = measure(none, test, ppl_tst, system=cfg["system_prompt_sv"])
    results["test"]["prompt_sv"] = r

    if best_edit:
        d = direction(best_edit["center"])[0]
        # The real weight edit with the dev-selected layers and strength.
        r, examples["edit"] = measure(
            lambda: weight_edit(model, best_edit["layers"], d, best_edit["lam"]), test, ppl_tst)
        results["test"]["edit"] = r
        # Control 3: random directions, same layers and strength. If the real edit does not
        # beat these, its effect is generic weight perturbation, not the Swedish direction.
        rand = []
        for s in range(cfg["n_random"]):
            g = torch.Generator().manual_seed(seed + 1 + s)
            rd = torch.randn(hidden, generator=g)
            rd = rd / rd.norm()
            rr, _ = measure(
                lambda: weight_edit(model, best_edit["layers"], rd, best_edit["lam"]), test, ppl_tst)
            rand.append(rr)
        results["test"]["random_edit"] = {
            "sv_rate_mean": float(np.mean([x["sv_rate"] for x in rand])),
            "sv_rate_sd": float(np.std([x["sv_rate"] for x in rand])),
            "ppl_sv_mean": float(np.mean([x["ppl_sv"] for x in rand])),
            "ppl_en_mean": float(np.mean([x["ppl_en"] for x in rand])),
            "runs": rand}

    if best_steer:
        d, nrm = direction(best_steer["center"])
        vec = best_steer["alpha"] * nrm * d
        # Conventional alternative: additive steering with the dev-selected layer/strength.
        r, examples["steer"] = measure(
            lambda: steering(model, best_steer["center"], vec), test, ppl_tst)
        results["test"]["steer"] = r

    # ---- save outputs ----------------------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    json.dump(results, open(OUT / "results.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False, default=float)
    with open(OUT / "examples.md", "w", encoding="utf-8") as f:
        f.write("# Example outputs (first 10 test prompts, outputs truncated to 300 chars)\n\n")
        for i, p in enumerate(test[:10]):
            f.write(f"## {i + 1}. {p}\n")
            for name, outs in examples.items():
                f.write(f"- **{name}**: {outs[i][:300].strip()!r}\n")
            f.write("\n")
    print("done. See results/results.json and results/examples.md")


if __name__ == "__main__":
    main()
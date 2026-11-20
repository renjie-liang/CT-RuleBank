"""CRIMSON and FineRadScore on the validation drafts (RJ 2026-11-08: look at their distributions on CT reports before deciding whether they enter the paper).
<out>/<gen>/crimson.jsonl and <out>/<gen>/fineradscore.jsonl, one line per draft, null where the metric gave no usable answer. For a bank: <out>/<gen>/<bank>/{crimson,fineradscore}.jsonl, the changed reports only; an unchanged report keeps its draft score."""
import json
import os
import tempfile

import numpy as np

from rulebank import evaluate as E
from rulebank.apply import changed, draft_texts
from rulebank.evidence import VALID_SPLIT, DraftSet, green_references
from rulebank.report import segment

CRIMSON_WEIGHTS = "/orange/xujie/liang.renjie/DATA/weights/medgemma-4b-it-crimson"


def crimson_weights():
    """the weights without the stale model.safetensors.index.json, which names shards the folder does not hold (eval/crimson/utils.py resolve_model_for_vllm handles only hub names)"""
    d = tempfile.mkdtemp(prefix="crimson_weights_")
    for f in os.listdir(CRIMSON_WEIGHTS):
        if f != "model.safetensors.index.json":
            os.symlink(os.path.join(CRIMSON_WEIGHTS, f), os.path.join(d, f))
    return d


def drafts_and_references(gen):
    refs = green_references(gen, VALID_SPLIT)
    texts = draft_texts(gen)
    return sorted(texts), texts, refs


def score_draft_crimson(gen, out):
    from eval.crimson import CRIMSONScore
    d = os.path.join(out, gen)
    os.makedirs(d, exist_ok=False)
    vids, texts, refs = drafts_and_references(gen)
    res = CRIMSONScore(api="vllm", model_name=crimson_weights()).evaluate_batch([refs[v] for v in vids], [texts[v] for v in vids])
    with open(os.path.join(d, "crimson.jsonl"), "x") as f:
        for v, r in zip(vids, res, strict=True):
            f.write(json.dumps({"vid": v, "result": r}) + "\n")
    print(f"[crimson] {gen}: {len(vids)} drafts, {sum(r is None for r in res)} without a usable answer", flush=True)


def score_draft_fineradscore(gen, out, base_url):
    from eval.fineradscore import score
    d = os.path.join(out, gen)
    os.makedirs(d, exist_ok=False)
    vids, texts, refs = drafts_and_references(gen)
    sentences = [[p["text"] for p in segment(texts[v]) if p["kind"] == "sent"] for v in vids]
    res = score([refs[v] for v in vids], sentences, base_url, workers=32, retries=2)
    with open(os.path.join(d, "fineradscore.jsonl"), "x") as f:
        for v, r in zip(vids, res, strict=True):
            f.write(json.dumps({"vid": v, "result": r}) + "\n")
    print(f"[fineradscore] {gen}: {len(vids)} drafts, {sum(r is None for r in res)} without a usable answer", flush=True)


def score_bank_crimson(gen, out, bank):
    from eval.crimson import CRIMSONScore
    R = changed(os.path.join(out, gen), bank)
    refs = green_references(gen, VALID_SPLIT)
    vids = sorted(R)
    res = CRIMSONScore(api="vllm", model_name=crimson_weights()).evaluate_batch([refs[v] for v in vids], [R[v]["text"] for v in vids])
    with open(os.path.join(out, gen, bank, "crimson.jsonl"), "x") as f:
        for v, r in zip(vids, res, strict=True):
            f.write(json.dumps({"vid": v, "result": r}) + "\n")
    print(f"[crimson] {gen} {bank}: {len(vids)} revised reports, {sum(r is None for r in res)} without a usable answer", flush=True)


def score_bank_fineradscore(gen, out, bank, base_url):
    from eval.fineradscore import score
    R = changed(os.path.join(out, gen), bank)
    refs = green_references(gen, VALID_SPLIT)
    vids = sorted(R)
    sentences = [[p["text"] for p in segment(R[v]["text"]) if p["kind"] == "sent"] for v in vids]
    res = score([refs[v] for v in vids], sentences, base_url, workers=32, retries=2)
    with open(os.path.join(out, gen, bank, "fineradscore.jsonl"), "x") as f:
        for v, r in zip(vids, res, strict=True):
            f.write(json.dumps({"vid": v, "result": r}) + "\n")
    print(f"[fineradscore] {gen} {bank}: {len(vids)} revised reports, {sum(r is None for r in res)} without a usable answer", flush=True)


def _results(path):
    return {x["vid"]: x["result"] for x in map(json.loads, open(path))}


def _paired(ids, draft, revised, value):
    """per report in ids: (draft value, revised value), nan where a metric gave no answer; an unchanged report keeps its draft value"""
    d, r = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    for i, v in enumerate(ids):
        if draft[v] is None or (v in revised and revised[v] is None):
            continue
        d[i] = value(draft[v])
        r[i] = value(revised[v]) if v in revised else d[i]
    return d, r


def summarise(gen, out, bank, draft_crimson, draft_fine):
    """-> <out>/<gen>/<bank>/extra_metrics.md: over all reports, paired, reports where either side has no answer left out; 200 patient bootstraps"""
    ids = list(DraftSet(gen, VALID_SPLIT).ids)
    B = E.patient_bootstraps(ids, 200, np.random.default_rng(0))
    d = os.path.join(out, gen, bank)
    urgent = lambda c: c["Emergent"] + c["Urgent"]
    total = lambda c: sum(c[k] for k in ("Emergent", "Urgent", "Actionable nonurgent", "Not actionable"))
    rows = [("CRIMSON x 100", draft_crimson, "crimson.jsonl", lambda r: 100 * r["crimson_score"]),
            ("FineRadScore urgent or emergent errors per report", draft_fine, "fineradscore.jsonl", lambda r: urgent(r["counts"])),
            ("FineRadScore all errors per report", draft_fine, "fineradscore.jsonl", lambda r: total(r["counts"]))]
    L = [f"# CRIMSON and FineRadScore, {gen}, bank {bank}", "", "| metric | reports | draft | revised | difference [95% CI] | changed reports, difference |", "|---|---|---|---|---|---|"]
    for name, draft_dir, file, value in rows:
        rev = _results(os.path.join(d, file))
        a, b = _paired(ids, _results(os.path.join(draft_dir, gen, file)), rev, value)
        keep = ~np.isnan(a)
        boot = np.array([np.nanmean((b - a)[x]) for x in B])
        ch = np.array([v in rev for v in ids]) & keep
        L.append(f"| {name} | {keep.sum()} | {a[keep].mean():.2f} | {b[keep].mean():.2f} | {(b - a)[keep].mean():+.2f} [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}] | {(b - a)[ch].mean():+.2f} (n {ch.sum()}) |")
    open(os.path.join(d, "extra_metrics.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

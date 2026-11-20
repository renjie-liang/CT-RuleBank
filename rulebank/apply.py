"""A learned bank on new drafts (the validation scans), and its four metrics. Stages 3-4 -> <out>/<gen>/: banks.json, summary.md,
<bank>/{texts.jsonl, fired.json, finebench_input.jsonl, radbert.npz, green.jsonl}, finebench/<bank>/."""
import collections
import json
import os

import numpy as np
import pandas as pd

from rulebank import ci as C
from rulebank.edits import possible_edits
from rulebank.evidence import DRAFT_SEED, VALID_SPLIT, DraftSet, draft_dir, green_references
from rulebank.gate import base_edit
from rulebank.labels import macro_f1
from rulebank.paths import V5
from rulebank.prompts import do_text
from rulebank.report import segment

ORDER = ["remove", "rewrite_normal", "side", "add_typed", "add", "add_value", "type"]
ADD_FAMILY = {"add", "add_typed"}     # one addition per (draft, abnormality), add_typed first: as evaluate.fired
DRAFT_FINEBENCH = V5 + "/results/finebench/valid1564_2026-10-29"


def draft_finebench(gen):
    """CT-FineBench of the validation drafts, per report"""
    return f"{draft_dir(gen, VALID_SPLIT)}/finebench/draft/per_report.csv" if DRAFT_SEED else f"{DRAFT_FINEBENCH}/{gen}_draft/per_report.csv"


def structured_rule(n, r):
    """a learned rule in the bank's format: when_code for code, do for the editor, why and record for people"""
    return {"id": f"R{n + 1}", "finding": r["finding"], "edit": r["edit"], "when_code": [[z, op, q] for z, op, _, q in r["conds"]],
            "do": do_text(r["finding"], r["edit"]), "why": r["reason"],
            "record": {k: r["outcome"][k] for k in ("n", "dF1", "dG", "precision", "agree") if k in r["outcome"]}}


def bank_document(name, rules):
    """the bank as people read it"""
    L = [f"# Rule bank {name} ({len(rules)} rules; generated, do not edit)", ""]
    for r in rules:
        when = " and ".join(f"{z} {op} {q:g}" for z, op, q in r["when_code"]) or "every eligible draft"
        L += [f"## {r['id']} · {r['finding']} · {r['edit']}", "", f"- When (code): {when}", f"- Do: {r['do']}", f"- Why: {r['why']}", f"- Record: {r['record']}", ""]
    return "\n".join(L)


def collect_banks(gen, runs, out):
    """runs: ['name=dir', ...]: the bank each run learned on all development drafts -> banks.json (the source), bank_<name>.md (for people)"""
    d = os.path.join(out, gen)
    os.makedirs(d, exist_ok=False)
    B = {}
    for spec in runs:
        nm, run = spec.split("=")
        run, source = run.split("@") if "@" in run else (run, gen)     # 'name=dir@other': the bank learned for the other generator (tab:model-specificity, RJ 2026-11-09)
        r = json.load(open(os.path.join(run, source, "run_final.json")))
        B[nm] = [structured_rule(n, r["rules"][i]) for n, i in enumerate(r["banks"][-1])]
        open(os.path.join(d, f"bank_{nm}.md"), "w").write(bank_document(nm, B[nm]))
    json.dump(B, open(os.path.join(d, "banks.json"), "w"), indent=1)
    print({k: len(v) for k, v in B.items()})


def rule_fires(r, ev, k):
    return all(ev[z][k] == ev[z][k] and (ev[z][k] >= q if op == ">=" else ev[z][k] <= q) for z, op, q in r["when_code"])


def align(drafts, v, text):
    """rules run one after another: per-sentence RadBERT of the running text, keyed by ITS sentence ids (sentences of the original draft
    keep their probabilities, a sentence a rule wrote gets zeros = states nothing)"""
    m = {}
    for p in segment(drafts.original[v]):
        if p["kind"] == "sent" and p["text"] not in m:
            m[p["text"]] = drafts.sentence_probs[v][p["id"]]
    return {p["id"]: m.get(p["text"], np.zeros(18, np.float32)) for p in segment(text) if p["kind"] == "sent"}


def revise_drafts(gen, out):
    d = os.path.join(out, gen)
    B = json.load(open(os.path.join(d, "banks.json")))
    drafts = DraftSet(gen, VALID_SPLIT)
    ev = drafts.evidence(sorted({c[0] for rules in B.values() for r in rules for c in r["when_code"]}))

    for name, rules in B.items():
        os.makedirs(os.path.join(d, name), exist_ok=False)
        rules = sorted(rules, key=lambda r: ORDER.index(r["edit"]))
        out_rows, n_fire = [], collections.Counter()
        for k, v in enumerate(drafts.ids):
            text = drafts.original[v]
            fired, added = [], set()
            for r in rules:
                if (r["edit"] in ADD_FAMILY and r["finding"] in added) or not rule_fires(r, ev, k):
                    continue
                probs = align(drafts, v, text)
                cand = [t for f, e, t in possible_edits(drafts, k, text, probs) if f == r["finding"] and base_edit(e) == r["edit"] and t is not None]
                if cand:
                    text = cand[0]
                    if r["edit"] in ADD_FAMILY:
                        added.add(r["finding"])
                    fired.append(f"{r['finding']}|{r['edit']}")
                    n_fire[f"{r['finding']}|{r['edit']}"] += 1
            if text != drafts.original[v]:
                out_rows.append({"vid": v, "text": text, "fired": fired})
        with open(os.path.join(d, name, "texts.jsonl"), "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        json.dump({"changed": len(out_rows), "fired": dict(n_fire)}, open(os.path.join(d, name, "fired.json"), "w"), indent=1)
        print(name, len(rules), "rules ->", len(out_rows), "changed", dict(n_fire))


def changed(d, name):
    return {r["vid"]: r for r in map(json.loads, open(os.path.join(d, name, "texts.jsonl")))}


def score_radbert(gen, out):
    from eval.radbert import radbert_soft
    d = os.path.join(out, gen)
    for name in json.load(open(os.path.join(d, "banks.json"))):
        R = changed(d, name)
        vids = sorted(R)
        soft = radbert_soft([R[v]["text"] for v in vids], device="cuda", bs=64) if vids else np.zeros((0, 18))
        np.savez(os.path.join(d, name, "radbert.npz"), vid=np.array(vids), soft=soft)
        print(name, len(vids))


def draft_texts(gen):
    return {x["vid"]: x["text"] for x in map(json.loads, open(f"{draft_dir(gen, VALID_SPLIT)}/drafts.jsonl"))}


def score_green(gen, out, name):
    """paired GREEN: draft and revised text of every changed report, scored in the same run"""
    from eval.green import score
    d = os.path.join(out, gen)
    R = changed(d, name)
    refs = green_references(gen, VALID_SPLIT)
    texts = draft_texts(gen)
    vids = sorted(R)
    res = score([refs[v] for v in vids] * 2, [texts[v] for v in vids] + [R[v]["text"] for v in vids])
    with open(os.path.join(d, name, "green.jsonl"), "x") as f:
        for i, v in enumerate(vids):
            f.write(json.dumps({"vid": v, "draft": res[i][0], "draft_analysis": res[i][1], "revised": res[len(vids) + i][0],
                                "revised_analysis": res[len(vids) + i][1]}) + "\n")


def score_all_draft_green(gen, out):
    """GREEN of every validation draft -> <out>/<gen>/green.jsonl. The absolute GREEN over all reports reads it: an unchanged report keeps its draft score (RJ 2026-11-08)"""
    from eval.green import score
    d = os.path.join(out, gen)
    os.makedirs(d, exist_ok=False)
    refs = green_references(gen, VALID_SPLIT)
    texts = draft_texts(gen)
    vids = sorted(texts)
    res = score([refs[v] for v in vids], [texts[v] for v in vids])
    with open(os.path.join(d, "green.jsonl"), "x") as f:
        for v, (g, analysis) in zip(vids, res, strict=True):
            f.write(json.dumps({"vid": v, "draft": g, "draft_analysis": analysis}) + "\n")


def write_finebench_inputs(gen, out):
    d = os.path.join(out, gen)
    texts = draft_texts(gen)
    for name in json.load(open(os.path.join(d, "banks.json"))):
        R = changed(d, name)
        with open(os.path.join(d, name, "finebench_input.jsonl"), "x") as f:
            for v in sorted(texts):
                f.write(json.dumps({"vid": v, "text": R[v]["text"] if v in R else texts[v]}) + "\n")


def summarise(gen, out):
    from eval.crg import crg
    d = os.path.join(out, gen)
    drafts = DraftSet(gen, VALID_SPLIT)
    ids = list(drafts.ids)
    Y, D = drafts.reference_labels, drafts.labels
    ci = lambda x: f"[{x[0]:+.2f}, {x[1]:+.2f}]"
    fd = pd.read_csv(draft_finebench(gen)).set_index("report_id")
    L = [f"# Learned banks on {VALID_SPLIT}, {gen} (rulebank.apply; banks learned on all development drafts, combined texts)", "",
         "| bank | rules | changed | dF1 [95% CI] | dCRG [95% CI] | dGREEN on changed [95% CI] | dCT-FineBench on changed with questions |",
         "|---|---|---|---|---|---|---|"]
    pos = {v: k for k, v in enumerate(ids)}
    for name, rules in json.load(open(os.path.join(d, "banks.json"))).items():
        R = changed(d, name)
        z = np.load(os.path.join(d, name, "radbert.npz"))
        P = D.copy()
        for v, s in zip(z["vid"].tolist(), z["soft"]):
            P[pos[v]] = s >= 0.5
        I = C.detection(D, P, Y, ids)
        g = "pending"
        gp = os.path.join(d, name, "green.jsonl")
        if os.path.exists(gp):
            G = [x for x in map(json.loads, open(gp)) if x["draft"] is not None and x["revised"] is not None
                 and "not provided" not in x["draft_analysis"] and "not provided" not in x["revised_analysis"]]
            dg = np.array([100 * (x["revised"] - x["draft"]) for x in G])
            g = f"{dg.mean():+.2f} {ci(C.mean_interval(dg, [x['vid'] for x in G]))} (n {len(dg)})"
        fb = "pending"
        fp = os.path.join(d, "finebench", name, "per_report.csv")
        if os.path.exists(fp):
            fe = pd.read_csv(fp).set_index("report_id")
            q = [r for r in fd.index[fd.n_qa > 0] if r in R]
            dq = 100 * (fe.score.loc[q] - fd.score.loc[q]).to_numpy()
            fb = f"{dq.mean():+.2f} {ci(C.mean_interval(dq, q))} (n {len(q)})" if q else "-"
        L.append(f"| {name} | {len(rules)} | {len(R)} | {macro_f1(P, Y) - macro_f1(D, Y):+.2f} {ci(I['f1']['diff'])} | {100 * (crg(P, Y) - crg(D, Y)):+.2f} {ci(I['crg']['diff'])} | {g} | {fb} |")
    open(os.path.join(d, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

"""The real editor at every learning round (RJ 2026-11-08). Learning does not change: its feedback stays the code-simulated edit outcomes. Afterwards the bank of each round, passed through the review, is executed by a real editor on the validation drafts and scored with RadBERT.
1. review_rounds: rules that left the bank before the last round were not reviewed; the same reviewer reads them, verdicts of the bank review are reused -> <reviewed>/<gen>/rounds.json
2. edit_rounds: an editor prompt depends only on the report and the rules that fired on it, so a report whose fired rules did not change between rounds is answered once. seed: the trace of the ordinary stage 3 run, so the last round repeats the reported result -> <out>/<gen>/curve/{texts.jsonl, rounds.jsonl, trace.jsonl}
3. score_rounds -> <out>/<gen>/curve/{radbert.npz, curve.csv, curve.md}; 200 patient bootstraps"""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from rulebank import editor as ED
from rulebank import ci as C
from rulebank import learner as L
from rulebank import prompts as P
from rulebank import review as RV
from rulebank.apply import ORDER, structured_rule
from rulebank.edits import possible_edits
from rulebank.evidence import VALID_SPLIT, DraftSet
from rulebank.labels import macro_f1
from rulebank.llm import Caller, FileCaller
from rulebank.report import numbered, segment


def _caller(base_url, exchange, trace, cfg):
    assert (base_url is None) != (exchange is None), "exactly one of base_url, exchange"
    return FileCaller(exchange, trace, cfg) if exchange else Caller(base_url, trace, cfg)


def review_rounds(gen, cfg, run, reviewed, base_url, exchange):
    """the bank after each round of the final run, without the rules the review refused"""
    r = json.load(open(os.path.join(run, gen, "run_final.json")))
    d = os.path.join(reviewed, gen)
    after = json.load(open(os.path.join(d, "run_final.json")))
    assert len(r["banks"]) == cfg.learner.rounds, (len(r["banks"]), cfg.learner.rounds)
    task = L.LearningTask(gen, cfg)
    rules = RV.without_selection(task, cfg, dict(r, banks=[sorted({i for b in r["banks"] for i in b})]))
    verdict = {}
    for x in map(json.loads, open(os.path.join(d, "trace.jsonl"))):                 # a retry comes after its failed try
        verdict[x["prompt"]] = x["answer"]
    asked = [x for x in rules.values() if not x.get("refused_without_condition") and RV.reviewed(x)]
    new = sorted({P.review_text(x) for x in asked} - set(verdict))
    caller = _caller(base_url, exchange, os.path.join(d, "trace_rounds.jsonl"), cfg.llm)

    def ask(item):
        n, text = item
        rec = caller.try_call(f"review_round|{n}", P.review_system(cfg), text, RV.SCHEMA, cfg.llm.temperature)
        if rec is None:
            rec = caller.try_call(f"review_round|{n}|retry1", P.review_system(cfg), text, RV.SCHEMA, cfg.llm.temperature)
        assert rec is not None, f"no usable review for: {text}"
        return rec["answer"]
    with ThreadPoolExecutor(16) as ex:
        verdict.update(zip(new, ex.map(ask, enumerate(new))))
    if exchange:
        caller.finish()

    def kept(i):
        x = rules[i]
        return not x.get("refused_without_condition") and (not RV.reviewed(x) or verdict[P.review_text(x)]["linked"])
    banks = [[i for i in b if kept(i)] for b in r["banks"]]
    assert banks[-1] == after["banks"][-1], "the last round must be the reviewed final bank"
    out = [[structured_rule(n, rules[i]) for n, i in enumerate(b)] for b in banks]
    json.dump(out, open(os.path.join(d, "rounds.json"), "w"), indent=1)
    print(f"[rounds] {gen}: {len(rules)} rules over the rounds, {len(new)} reviewed in this step, sizes {[len(b) for b in banks]}", flush=True)


def _key(gen, v, prompt):
    return f"{gen}|curve|{v}|{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"


def _seed(gen, seed_trace, trace):
    """the answers of the ordinary stage 3 run, under the keys of this step"""
    with open(trace, "x") as f:
        for x in map(json.loads, open(seed_trace)):
            parts = x["key"].split("|")
            assert parts[0] == gen and len(parts) in (3, 4), x["key"]
            f.write(json.dumps(dict(x, key=_key(gen, parts[2], x["prompt"]) + "".join(f"|{p}" for p in parts[3:]))) + "\n")


def edit_rounds(gen, reviewed, out, cfg, seed_trace, base_url, exchange, limit):
    """limit > 0: only the first drafts of the validation set, for a smoke test"""
    d = os.path.join(out, gen, "curve")
    os.makedirs(d, exist_ok=False)
    rounds = json.load(open(os.path.join(reviewed, gen, "rounds.json")))
    drafts = DraftSet(gen, VALID_SPLIT)
    ev = drafts.evidence(sorted({c[0] for rules in rounds for r in rules for c in r["when_code"]}))
    trace = os.path.join(d, "trace.jsonl")
    _seed(gen, seed_trace, trace)
    ids = list(drafts.ids)[:limit] if limit else list(drafts.ids)
    eligible = {v: possible_edits(drafts, k, drafts.original[v], drafts.sentence_probs[v]) for k, v in enumerate(drafts.ids) if v in set(ids)}
    pos = {v: k for k, v in enumerate(drafts.ids)}
    calls, per_round = {}, []
    for rules in rounds:
        rules = sorted(rules, key=lambda r: ORDER.index(r["edit"]))
        jobs = []
        for v in ids:
            k = pos[v]
            fired = ED.fired_on(rules, ev, k, eligible[v])
            if fired:
                pieces = segment(drafts.original[v])
                text = "\n".join(f"{i}: {ED.instruction(drafts, k, r, e)}" for i, r, e in fired)
                prompt = P.editor_prompt(numbered(pieces), ED.evidence_sheet(fired, ev, k), text)
                key = _key(gen, v, prompt)
                calls[key] = (prompt, [p["id"] for p in pieces if p["kind"] == "sent"], [i for i, _, _ in fired])
                jobs.append((v, key))
        per_round.append(jobs)
    caller = _caller(base_url, exchange, trace, cfg)
    print(f"[curve] {gen}: {sum(map(len, per_round))} report-rounds, {len(calls)} distinct prompts, {sum(k in caller.done for k in calls)} from the seed trace", flush=True)
    keys = sorted(calls)
    with ThreadPoolExecutor(cfg.workers) as ex:
        answers = dict(zip(keys, ex.map(lambda k: ED.ask(caller, k, *calls[k], cfg, P.EDITOR_SYSTEM), keys)))
    if exchange:
        caller.finish()
    with open(os.path.join(d, "texts.jsonl"), "x") as ft, open(os.path.join(d, "rounds.jsonl"), "x") as fr:
        for n, jobs in enumerate(per_round):
            failed, changed = 0, 0
            for v, key in jobs:
                a = answers[key]
                failed += a is None
                if a is not None and a["edits"]:
                    text, _ = ED.apply_edits(drafts.original[v], a, gen)
                    changed += 1
                    ft.write(json.dumps({"round": n + 1, "vid": v, "text": text}) + "\n")
            fr.write(json.dumps({"round": n + 1, "rules": len(rounds[n]), "calls": len(jobs), "failed": failed, "changed": changed}) + "\n")
    print(f"[curve] {gen}: done, {sum(a is None for a in answers.values())} prompts without a usable answer", flush=True)


def score_rounds(gen, out):
    from eval.crg import crg
    from eval.radbert import radbert_soft
    d = os.path.join(out, gen, "curve")
    rows = list(map(json.loads, open(os.path.join(d, "texts.jsonl"))))
    distinct = sorted({x["text"] for x in rows})
    saved = os.path.join(d, "radbert.npz")
    if os.path.exists(saved):                                            # already scored; only the summaries are written again
        z = np.load(saved, allow_pickle=True)
        assert z["text"].tolist() == distinct
        soft = z["soft"]
    else:
        soft = radbert_soft(distinct, device="cuda", bs=64)
        np.savez(saved, text=np.array(distinct, dtype=object), soft=soft)
    at = {t: i for i, t in enumerate(distinct)}
    drafts = DraftSet(gen, VALID_SPLIT)
    ids = list(drafts.ids)
    pos = {v: k for k, v in enumerate(ids)}
    Y, D = drafts.reference_labels, drafts.labels
    info = [json.loads(x) for x in open(os.path.join(d, "rounds.jsonl"))]
    table = []
    for x in info:
        Pred = D.copy()
        for r in rows:
            if r["round"] == x["round"]:
                Pred[pos[r["vid"]]] = soft[at[r["text"]]] >= 0.5
        lo, hi = C.detection(D, Pred, Y, ids)["f1"]["diff"]
        table.append(dict(x, F1=macro_f1(Pred, Y), dF1=macro_f1(Pred, Y) - macro_f1(D, Y), dF1_lo=lo, dF1_hi=hi,
                          CRG=100 * crg(Pred, Y), dCRG=100 * (crg(Pred, Y) - crg(D, Y))))
    df = pd.DataFrame(table)
    df.to_csv(os.path.join(d, "curve.csv"), index=False)
    L_ = [f"# Real editor at every learning round, {gen}, draft macro F1 {macro_f1(D, Y):.2f}", "", "| round | rules | calls | failed | changed | F1 | dF1 [95% CI] | CRG | dCRG |", "|---|---|---|---|---|---|---|---|---|"]
    L_ += [f"| {x['round']} | {x['rules']} | {x['calls']} | {x['failed']} | {x['changed']} | {x['F1']:.2f} | {x['dF1']:+.2f} [{x['dF1_lo']:+.2f}, {x['dF1_hi']:+.2f}] | {x['CRG']:.2f} | {x['dCRG']:+.2f} |" for x in table]
    open(os.path.join(d, "curve.md"), "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))

"""The review of a learned bank (RJ 2026-11-07), after the gate and before the bank is applied. Learning does not change and the agent never sees the review.
1. Code: a condition whose threshold selects at least 95% of the eligible learning drafts selects nothing (often a volume >= 0, where most scans have none). The rule loses its condition, says so
   ("every eligible draft"), and is scored again; if the gate refuses it then, it leaves the bank.
2. Rules without a condition and the edits that carry their own image evidence (side, type, add_typed) are not reviewed (RJ 2026-11-07): there is no
   evidence-finding link to judge, the gate alone decides. The agent reads every other rule without its scores and without its threshold, says why it should hold or fail, and decides whether a radiologist would see a medical link between the evidence and the finding in that
   direction. Each (finding, edit, evidence, direction) is read once and its verdict holds in every fold (RJ 2026-11-08). A rule without a link leaves the bank; the reason becomes the rule's why.
Every outer fold's bank is reviewed, so the held-out estimate includes the review."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rulebank import evaluate as E
from rulebank import learner as L
from rulebank import prompts as P
from rulebank.llm import Caller, FileCaller, json_schema

FOLDS = ["0", "1", "2", "3", "4", "final"]
NON_SELECTING = 0.95
NOT_REVIEWED = {"linked": True, "why": "not reviewed: no condition, or an edit that carries its own image evidence"}
SCHEMA = json_schema("review", {"type": "object", "additionalProperties": False, "required": ["why", "linked"],
                                "properties": {"why": {"type": "string"}, "linked": {"type": "boolean"}}})


def reviewed(x):
    """whether the agent judges this rule: it has a condition and its edit does not carry its own evidence"""
    return bool(x["conds"]) and x["edit"] not in P.NO_EVIDENCE_EDITS


def without_selection(task, cfg, r):
    """step 1 on one run: {rule id: the rule as it enters the review}"""
    all_drafts = set(task.drafts.ids)
    learn = all_drafts - set(r["apply_to"]) if r["apply_to"] is not None else all_drafts
    agent = L.RuleAgent(task, cfg, learn)
    out = {}
    for i in r["banks"][-1]:
        x = r["rules"][i]
        eligible = task.table.eligible_rows[(x["finding"], x["edit"])]
        if x["conds"] and len(L.rebuilt(task, x, learn)["rows"]) >= NON_SELECTING * len(eligible):
            x = dict(x, conds=[], condition_dropped=x["conds"])
            rule = L.rebuilt(task, x, learn)
            x["outcome"] = agent.score(rule)
            x["refused_without_condition"] = agent.refused(rule, x["outcome"])
        out[i] = x
    return out


def review(gen, cfg, run, out_root, base_url, exchange):
    """-> <out_root>/<name>_reviewed/<gen>/{run_<fold>.json, review.jsonl, summary.md, trace.jsonl}. Exactly one of base_url (the local LLM) and exchange (rulebank.file_exchange: Claude)"""
    assert (base_url is None) != (exchange is None), "exactly one of base_url, exchange"
    name = os.path.basename(os.path.normpath(run))
    d = os.path.join(out_root, f"{name}_reviewed", gen)
    os.makedirs(d, exist_ok=False)
    task = L.LearningTask(gen, cfg)
    runs = {f: json.load(open(os.path.join(run, gen, f"run_{f}.json"))) for f in FOLDS}
    rules = {f: without_selection(task, cfg, r) for f, r in runs.items()}
    texts = sorted({P.review_text(x) for f in FOLDS for x in rules[f].values() if not x.get("refused_without_condition") and reviewed(x)})
    trace = os.path.join(d, "trace.jsonl")
    caller = FileCaller(exchange, trace, cfg.llm) if exchange else Caller(base_url, trace, cfg.llm)

    def ask(item):
        n, text = item
        rec = caller.try_call(f"review|{n}", P.review_system(cfg), text, SCHEMA, cfg.llm.temperature)
        if rec is None:
            rec = caller.try_call(f"review|{n}|retry1", P.review_system(cfg), text, SCHEMA, cfg.llm.temperature)
        assert rec is not None, f"no usable review for: {text}"
        return rec["answer"]
    with ThreadPoolExecutor(16) as ex:
        verdict = dict(zip(texts, ex.map(ask, enumerate(texts))))
    if exchange:
        caller.finish()
    log = []
    for f, r in runs.items():
        kept = []
        for i, x in rules[f].items():
            if x.get("refused_without_condition"):
                v = {"linked": False, "why": f"without its condition the gate refuses it: {x['refused_without_condition']}"}
            else:
                v = verdict[P.review_text(x)] if reviewed(x) else NOT_REVIEWED
            r["rules"][i] = dict(x, proposal_reason=x["reason"], reason=v["why"], linked=v["linked"])
            log.append({"fold": f, "id": i, "rule": P.rule_text(x), "condition_dropped": "condition_dropped" in x, "reviewed": reviewed(x), "linked": v["linked"], "why": v["why"]})
            if v["linked"]:
                kept.append(i)
        r["banks"] = [*r["banks"], kept]
        r["review"] = {"before": r["banks"][-2], "after": kept}
        json.dump(r, open(os.path.join(d, f"run_{f}.json"), "w"), indent=1)
    with open(os.path.join(d, "review.jsonl"), "w") as fh:
        for x in log:
            fh.write(json.dumps(x) + "\n")
    B = E.patient_bootstraps(task.drafts.ids, 200, np.random.default_rng(0))
    S = [f"# Review of {run}, {gen} (rulebank/review.py)", "", "| bank | rules per fold | held-out dF1 [95% CI] | held-out dGREEN [95% CI] | edits |", "|---|---|---|---|---|"]
    for label, which in (("as learned", -2), ("after the review", -1)):
        rows = L.held_out_rows(task, [dict(runs[f], banks=[runs[f]["banks"][which]]) for f in FOLDS[:5]], B)
        S.append(rows[0].replace("| 1 |", f"| {label} |", 1))
    fin = [x for x in log if x["fold"] == "final"]
    S += ["", f"Final bank: {len(fin)} rules, condition dropped {sum(x['condition_dropped'] for x in fin)}, no medical link {sum(not x['linked'] for x in fin)}.", "",
          "| rule | condition dropped | reviewed | linked | why |", "|---|---|---|---|---|"]
    S += [f"| {x['rule']} | {'yes' if x['condition_dropped'] else ''} | {'yes' if x['reviewed'] else ''} | {'yes' if x['linked'] else '**no**'} | {x['why'].replace('|', '/')} |" for x in fin]
    open(os.path.join(d, "summary.md"), "w").write("\n".join(S) + "\n")
    print("\n".join(S[:6]))

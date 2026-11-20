"""The rule agent. The bank starts with the list of evidence only; each round the LLM proposes probes, code tries each at 11 thresholds on the
development drafts and returns the revision outcomes, the gate decides what enters the bank, and the prompt keeps the outcomes as experience."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from rulebank import evaluate as E
from rulebank import gate
from rulebank import prompts as P
from rulebank.config import edits_off, evidence_names
from rulebank.evidence import DEV_SPLIT, DraftSet
from rulebank.labels import macro_f1
from rulebank.llm import Caller, FileCaller, ReplayCaller, json_schema
from rulebank.table import EditTable


class LearningTask:
    """what stays fixed while one generator's bank is learned: its drafts, the edit-outcome table and the evidence"""
    def __init__(self, gen, cfg):
        self.gen = gen
        self.drafts = drafts = DraftSet(gen, DEV_SPLIT[gen])
        self.table = EditTable(cfg.table, gen, drafts.ids)
        off = edits_off(cfg.evidence)
        self.table.eligible_rows = {k: v for k, v in self.table.eligible_rows.items() if k[1] not in off}
        self.edits = [e for e in P.EDIT_DOC if e not in off]
        self.evidence = drafts.evidence(evidence_names(cfg.evidence))
        self.anon_names = None
        if cfg.learner.anon:
            names = sorted(self.evidence)
            perm = np.random.default_rng(2026).permutation(len(names))
            self.anon_names = {names[i]: f"ev{k + 1:02d}" for k, i in enumerate(perm)}
            self.evidence = {self.anon_names[z]: self.evidence[z] for z in names}
        self.evidence_names = sorted(self.evidence)
        self.abnormalities = sorted({f for f, e in self.table.eligible_rows})


def schema(task, cfg):
    names = lambda v: {"enum": v} if cfg.learner.enum_names else {}
    rule = {"type": "object", "additionalProperties": False, "required": ["finding", "edit", "evidence", "direction", "reason"],
            "properties": {"finding": {"type": "string", **names(task.abnormalities)}, "edit": {"type": "string", "enum": task.edits},
                           "evidence": {"type": "string", **names(task.evidence_names + ([P.NO_EVIDENCE] if set(P.NO_EVIDENCE_EDITS) <= set(task.edits) else []))}, "direction": {"type": "string", "enum": [">=", "<="]},
                           "reason": {"type": "string"}}}
    return json_schema("rules", {"type": "object", "additionalProperties": False, "required": ["rules"], "properties": {"rules": {"type": "array", "items": rule}}})


def probe_to_rule(task, probe, learn, threshold):
    """a probe at a percentile threshold (None: no evidence) -> rule {finding, edit, conds: [(evidence, direction, percentile, value)], rows}, or a reject reason.
    The percentile is taken over the eligible drafts in learn; the rule fires on every eligible row."""
    f, e, z, op = probe["finding"], probe["edit"], probe["evidence"], probe["direction"]
    table = task.table
    if (f, e) not in table.eligible_rows:
        return P.REJECT["no_edit"].format(e=e, f=f)
    allk = table.eligible_rows[(f, e)]
    if z == P.NO_EVIDENCE:
        return {"finding": f, "edit": e, "conds": [], "rows": allk} if e in P.NO_EVIDENCE_EDITS else P.REJECT["no_evidence"]
    if z not in task.evidence_names:
        return P.REJECT["evidence"].format(names=task.evidence_names)
    value = lambda ks: np.array([task.evidence[z][table.position[table.rows[k]["vid"]]] for k in ks])
    x_fit = value([k for k in allk if table.rows[k]["vid"] in learn])
    ok = ~np.isnan(x_fit)
    if ok.sum() < 5:
        return P.REJECT["few_values"].format(evidence=z)
    q = float(np.percentile(x_fit[ok], float(threshold[1:])))
    x = value(allk)
    sel = ~np.isnan(x) & ((x >= q) if op == ">=" else (x <= q))
    return {"finding": f, "edit": e, "conds": [(z, op, threshold, round(q, 4))], "rows": [k for k, s in zip(allk, sel) if s]}


def rule_key(rule):
    return (rule["finding"], rule["edit"], tuple((c[0], c[1], c[2]) for c in rule["conds"]))


def signature(probe):
    return probe["finding"], probe["edit"], probe["evidence"], probe["direction"]


class RuleAgent:
    """one run of the agent on the drafts in learn: the probes tried, the rules scored, the bank after each round"""
    def __init__(self, task, cfg, learn):
        self.task, self.cfg, self.learn = task, cfg, learn
        members = [task.table.position[v] for v in learn]
        self.base_f1 = macro_f1(task.drafts.labels[members], task.drafts.reference_labels[members])
        self.min_n = gate.min_changed(learn, len(task.drafts.ids), cfg.gate)
        self.probes, self.rules, self.banks, self.failed_rounds = [], {}, [], []
        self.tried, self.refused_repeats = set(), 0

    def score(self, rule):
        return gate.rule_outcome(rule, self.learn, self.task.table, self.task.drafts, self.base_f1)

    def refused(self, rule, outcome):
        return gate.refused(rule, outcome, self.min_n, self.cfg.gate)

    def experience(self, rnd):
        if self.cfg.learner.blind:
            return P.blind_experience_text(self.probes)
        active = []
        if not self.cfg.learner.no_experience and self.banks:
            by_id = {r["id"]: r for r in self.rules.values()}
            active = [(i, by_id[i], self.score(by_id[i])) for i in self.banks[-1]]
        return P.experience_text(self.task.evidence_names, self.probes, active, rnd, self.cfg.learner.no_experience)

    def run_probe(self, probe, rnd):
        """one probe of the LLM: code tries it at all thresholds, scores each and records it; the gate's verdict is shown, not yet applied"""
        pid = str(len(self.probes))
        record = {"id": pid, "round": rnd, "raw": probe, "sweep": []}
        for threshold in ([None] if probe["evidence"] == P.NO_EVIDENCE else P.THRESHOLDS):
            rule = probe_to_rule(self.task, probe, self.learn, threshold)
            if isinstance(rule, str):
                record["reject"] = rule
                break
            if rule_key(rule) in self.rules:
                record["sweep"].append({"threshold": threshold, "seen": True})
                continue
            outcome = self.score(rule)
            self.rules[rule_key(rule)] = dict(rule, id=f"{pid}.{threshold}" if threshold else pid, round=rnd, reason=probe["reason"], outcome=outcome)
            record["sweep"].append({"threshold": threshold, "value": rule["conds"][0][3] if rule["conds"] else None, "outcome": outcome, "refused": self.refused(rule, outcome)})
        self.probes.append(record)

    def ask(self, key, system, prompt, caller):
        """the LLM's probes, or None when the reply is unusable twice"""
        schema_, temperature = schema(self.task, self.cfg), self.cfg.llm.temperature
        rec = caller.try_call(key, system, prompt, schema_, temperature)
        if rec is None:                                         # cut off or not JSON: one retry
            rec = caller.try_call(f"{key}|retry1", system, prompt, schema_, temperature)
        return None if rec is None else rec["answer"]["rules"]

    def run_round(self, key, system, prompt, rnd, budget, caller):
        """up to budget new probes: those that repeat an earlier one are refused, and the LLM is asked for others up to repeat_retries times"""
        left, note = budget, ""
        for attempt in range(self.cfg.learner.repeat_retries + 1):
            reply = self.ask(key if attempt == 0 else f"{key}|more{attempt}", system, prompt + note, caller)
            if reply is None:
                self.failed_rounds.append(rnd)
                return
            repeats = []
            for probe in reply[:budget]:
                if left == 0:
                    break
                if self.cfg.learner.repeat_retries and signature(probe) in self.tried:
                    repeats.append(probe)
                    continue
                self.tried.add(signature(probe))
                self.run_probe(probe, rnd)
                left -= 1
            self.refused_repeats += len(repeats)
            if left == 0 or not repeats:
                return
            note = P.repeat_note(repeats, left)

    def run(self, fold, caller):
        """rounds of: prompt -> LLM -> probes -> the gate selects the bank"""
        task, learner = self.task, self.cfg.learner
        rounds, budget = learner.rounds, learner.budget
        edits_of = {f: sorted(e for g, e in task.table.eligible_rows if g == f) for f in task.abnormalities}
        head = P.grammar_text(task.evidence_names, edits_of, task.edits) + "\n\n" + P.evidence_text(task.evidence_names, learner.anon)
        system = P.system_prompt(self.cfg)
        for rnd in range(rounds):
            prompt = P.user_prompt(head, self.experience(rnd), rnd, rounds, budget, learner.prompt)
            self.run_round(f"{task.gen}|{fold}|round{rnd}", system, prompt, rnd, budget, caller)
            self.banks.append(gate.select_bank([x for x in self.rules.values() if x["round"] <= rnd], self.learn, task.table, task.drafts, self.cfg.gate))


def learn_bank(task, cfg, fold, learn, apply_to, caller):
    """the agent on the drafts in learn; apply_to = the outer fold held out from it (None for the bank on all development drafts)"""
    agent = RuleAgent(task, cfg, learn)
    agent.run(fold, caller)
    return {"fold": fold, "n_learn": len(learn), "probes": agent.probes, "rules": {r["id"]: {k: v for k, v in r.items() if k != "rows"} for r in agent.rules.values()},
            "banks": agent.banks, "failed_rounds": agent.failed_rounds, "refused_repeats": agent.refused_repeats, "apply_to": None if apply_to is None else sorted(apply_to)}


def rebuilt(task, x, learn):
    """a stored rule with its rows on the drafts in learn"""
    if x["conds"]:
        z, op, threshold, _ = x["conds"][0]
        return probe_to_rule(task, {"finding": x["finding"], "edit": x["edit"], "evidence": z, "direction": op}, learn, threshold)
    return {"finding": x["finding"], "edit": x["edit"], "conds": [], "rows": task.table.eligible_rows[(x["finding"], x["edit"])]}   # no evidence: also a rule the review freed of its condition


def held_out_rows(task, folds, B):
    """after each round: the pooled result of every fold's bank on its own held-out patients"""
    all_drafts, L, built = set(task.drafts.ids), [], {}
    for rnd in range(max(len(r["banks"]) for r in folds)):
        fired, sizes = {}, []
        if not any(r["banks"][rnd] for r in folds):
            L.append(f"| {rnd + 1} | {[0] * len(folds)} | +0.00 (empty bank = the draft) | - | 0 |")
            continue
        for r in folds:
            ids = r["banks"][rnd]
            sizes.append(len(ids))
            learn = all_drafts - set(r["apply_to"])
            rules = {}
            for i in ids:
                x = r["rules"][i]
                rules[(x["finding"], x["edit"])] = built.setdefault((r["fold"], i), rebuilt(task, x, learn))
            fired.update(E.fired_rows(task.table, rules, set(r["apply_to"])))
        ho = E.evaluate_edits(fired, task.table, task.drafts, B)
        L.append(f"| {rnd + 1} | {sizes} | {ho['dF1']:+.2f} [{ho['dF1_ci'][0]:+.2f}, {ho['dF1_ci'][1]:+.2f}] | {ho['dG']:+.2f} [{ho['dG_ci'][0]:+.2f}, {ho['dG_ci'][1]:+.2f}] | {ho['edits']} |")
    return L


def main(gen, cfg, out_root, folds, base_url, replay, exchange, random_from):
    """the bank on each outer fold (for the held-out estimate) and on all development drafts -> <out_root>/<name>/<gen>/{run_<fold>.json, config.yaml, summary.md, trace.jsonl}"""
    d = os.path.join(out_root, cfg.name, gen)
    os.makedirs(d, exist_ok=False)
    yaml.safe_dump(cfg.model_dump(), open(os.path.join(d, "config.yaml"), "w"), sort_keys=False)
    task = LearningTask(gen, cfg)
    if cfg.learner.anon:
        json.dump(task.anon_names, open(os.path.join(d, "anon_map.json"), "w"), indent=1)
    all_drafts = set(task.drafts.ids)
    outer = E.patient_folds(all_drafts, cfg.folds.outer, cfg.folds.seed)
    assert sum(x is not None for x in (base_url, replay, exchange, random_from)) == 1, "exactly one of base_url, replay, exchange, random_from"
    trace = os.path.join(d, "trace.jsonl")
    if random_from:
        from baselines.random_proposer import RandomCaller
        caller = RandomCaller(task, cfg, random_from, gen, trace)
    else:
        caller = ReplayCaller(replay, folds) if replay else FileCaller(exchange, trace, cfg.llm) if exchange else Caller(base_url, trace, cfg.llm)
    jobs = [(f, all_drafts - outer[int(f)], outer[int(f)]) if f != "final" else (f, all_drafts, None) for f in folds]
    with ThreadPoolExecutor(len(jobs)) as ex:
        runs = list(ex.map(lambda j: learn_bank(task, cfg, j[0], j[1], j[2], caller), jobs))
    for r in runs:
        json.dump(r, open(os.path.join(d, f"run_{r['fold']}.json"), "w"), indent=1)
    held = [r for r in runs if r["fold"] != "final"]
    L = [f"# Rule agent, {cfg.name}, {gen} ({cfg.learner.rounds} rounds x {cfg.learner.budget})", ""]
    if len(held) == 5:
        L += ["| after round | rules per fold | held-out dF1 [95% CI] | held-out dGREEN [95% CI] | edits |", "|---|---|---|---|---|"]
        L += held_out_rows(task, held, E.patient_bootstraps(task.drafts.ids, 200, np.random.default_rng(0)))
    for r in runs:
        L += ["", f"## run {r['fold']}: learned on {r['n_learn']} drafts; proposed {len(r['probes'])}, rejected {sum('reject' in h for h in r['probes'])}, "
                  f"valid {len(r['rules'])}; failed LLM rounds {r['failed_rounds']}; " + (f"refused repeats {r['refused_repeats']}; " if cfg.learner.repeat_retries else "") + f"final bank {len(r['banks'][-1])} rules", ""]
        L += [f"- [{i}] round {r['rules'][i]['round'] + 1}: {P.rule_text(r['rules'][i])} - {r['rules'][i]['reason']}" for i in r["banks"][-1]]
    open(os.path.join(d, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if replay:
        caller.check_all_used()
    if exchange:
        caller.finish()

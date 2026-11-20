"""Text memory-guided revision, a control for the Rule Bank (RJ 2026-11-08, tab:direct-revision). The whole learned bank is written as text in the editor's prompt, with each rule's
evidence condition, threshold, draft condition and instruction, next to the draft and the values of all the evidence the bank uses. The editor decides which rules apply: code does not look at
the conditions or at the draft's eligibility. The same edit list, checks and application as the Rule Bank's editor (rulebank/editor.py). Every draft gets a call.
Writes the same files as editor.edit_drafts; <bank>/editor.jsonl also holds the rules that code would have fired, for comparison only.
    MODULE=steps.stage3_apply_bank ARGS="--gen selfcue --runs text_memory=<reviewed run> --out <dir> --executor text_memory --editor configs/editor/gptoss.yaml" sbatch slurm/gptoss_run_compact.sbatch"""
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor

from rulebank import prompts as P
from rulebank.apply import ORDER
from rulebank.editor import apply_edits, fired_rules, problem, schema
from rulebank.edits.details import DEVICE, SIDE_SENT, value_sentence
from rulebank.evidence import VALID_SPLIT, DraftSet
from rulebank.llm import Caller
from rulebank.report import numbered, segment


def measured_instruction(drafts, k, rule):
    """the rule's do text with this scan's measured values, or None when a value it needs was not measured"""
    text, f = rule["do"], rule["finding"]
    if P.SIDE in text:
        if drafts.side[k] not in SIDE_SENT:
            return None
        text = text.replace(P.SIDE, drafts.side[k])
    if P.DEVICE_SENTENCE in text:
        found = [DEVICE[t][0] for t in sorted(drafts.device_types[k]) if t in DEVICE]
        if not found:
            return None
        text = text.replace(P.DEVICE_SENTENCE, " / ".join(found))
    if P.VALUE_SENTENCE in text:
        value = value_sentence(drafts, k, f)
        if value is None:
            return None
        text = text.replace(P.VALUE_SENTENCE, value)
    if P.ASCENDING_MM in text:
        mm = float(drafts.ascending_mm[k])
        if mm != mm:
            return None
        text = text.replace(P.ASCENDING_MM, str(round(mm)))
    return text


def rule_text(rule_id, rule, instruction):
    condition = " and ".join(f"{z} {op} {q:g}" for z, op, q in rule["when_code"]) or "none"
    do = instruction if instruction is not None else "this rule cannot apply to this scan: the value it needs was not measured."
    return f"{rule_id} ({rule['finding']}, {rule['edit']}). Evidence condition: {condition}. Draft condition: {P.EDIT_DOC[rule['edit']]}. Instruction: {do}"


def whole_sheet(names, ev, k):
    return "\n".join(f"- {z} ({P.evidence_line(z)}): {ev[z][k]:g}" if ev[z][k] == ev[z][k] else f"- {z} ({P.evidence_line(z)}): not available" for z in names)


def ask(caller, key, prompt, sentence_ids, rule_ids, cfg):
    """the checked answer; None when every try failed"""
    for t in range(cfg.retries + 1):
        rec = caller.try_call(key if t == 0 else f"{key}|retry{t}", P.TEXT_MEMORY_SYSTEM, prompt, schema(rule_ids, sentence_ids), cfg.temperature if t == 0 else cfg.retry_temperature)
        if rec is not None and problem(rec["answer"], sentence_ids) is None:
            return rec["answer"]
    return None


def edit_drafts(gen, out, cfg, base_url, limit):
    """every bank of <out>/<gen>/banks.json on the validation drafts; limit > 0: only the first drafts, for a smoke test"""
    d = os.path.join(out, gen)
    B = json.load(open(os.path.join(d, "banks.json")))
    drafts = DraftSet(gen, VALID_SPLIT)
    names = sorted({c[0] for rules in B.values() for r in rules for c in r["when_code"]})
    ev = drafts.evidence(names)
    caller = Caller(base_url, os.path.join(d, "trace.jsonl"), cfg)
    for name, rules in B.items():
        os.makedirs(os.path.join(d, name), exist_ok=False)
        rules = sorted(rules, key=lambda r: ORDER.index(r["edit"]))
        ids = [f"R{i + 1}" for i in range(len(rules))]
        jobs = list(range(len(drafts.ids)))[:limit] if limit else list(range(len(drafts.ids)))

        def one(k):
            v = drafts.ids[k]
            pieces = segment(drafts.original[v])
            sid = [p["id"] for p in pieces if p["kind"] == "sent"]
            text = "\n".join(rule_text(i, r, measured_instruction(drafts, k, r)) for i, r in zip(ids, rules))
            return ask(caller, f"{gen}|{name}|{v}", P.editor_prompt(numbered(pieces), whole_sheet(names, ev, k), text), sid, ids, cfg)

        with ThreadPoolExecutor(cfg.workers) as ex:
            answers = list(ex.map(one, jobs))
        rows, n_fire, failed, n_repeats, both, llm_only, code_only = [], collections.Counter(), 0, 0, 0, 0, 0
        label = {i: f"{r['finding']}|{r['edit']}" for i, r in zip(ids, rules)}
        with open(os.path.join(d, name, "editor.jsonl"), "w") as f:
            for k, a in zip(jobs, answers):
                v = drafts.ids[k]
                code = {f"{r['finding']}|{r['edit']}" for _, r, _ in fired_rules(rules, ev, drafts, k)}
                f.write(json.dumps({"vid": v, "code_would_fire": sorted(code), "answer": a}) + "\n")
                if a is None:
                    failed += 1
                    code_only += len(code)
                    continue
                used = {label[x["rule_id"]] for x in a["edits"]}
                both, llm_only, code_only = both + len(used & code), llm_only + len(used - code), code_only + len(code - used)
                if not a["edits"]:                               # no edit: the draft stays as it is
                    continue
                n_fire.update(sorted(used))
                text, repeats = apply_edits(drafts.original[v], a, gen)
                n_repeats += repeats
                rows.append({"vid": v, "text": text, "fired": sorted(used)})
        with open(os.path.join(d, name, "texts.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        json.dump({"changed": len(rows), "fired": dict(n_fire), "calls": len(jobs), "failed": failed, "cited_and_code_fires": both, "cited_code_would_not_fire": llm_only,
                   "code_fires_not_cited": code_only, "repeated_sentences_dropped": n_repeats}, open(os.path.join(d, name, "fired.json"), "w"), indent=1)
        print(name, len(rules), "rules:", len(jobs), "calls,", failed, "failed,", len(rows), "changed; cited and code fires", both, ", cited only", llm_only, ", code only", code_only, flush=True)

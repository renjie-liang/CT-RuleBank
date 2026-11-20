"""The editor agent: a learned bank on new drafts, executed by LLM + code. Code decides WHEN, as with the code edits: a rule fires when its conditions hold and the draft is eligible for the
edit. The LLM decides HOW: it reads the numbered draft, the evidence of the rules that fired and their instructions, and returns a list of edits, each naming its rule; code checks the
list and applies it. One call per draft with every rule that fired on it. From main_v5 stage_b/editor.py (LLM edit list), analysis/editor_v2_smoke.py (code WHEN, LLM HOW).
Writes the same files as apply.revise_drafts, plus <bank>/editor.jsonl (the answers) and <gen>/trace.jsonl."""
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor

from rulebank import prompts as P
from rulebank.apply import ADD_FAMILY, ORDER, rule_fires
from rulebank.edits import possible_edits, renumber
from rulebank.edits.details import DEVICE, VALUE_SITES, value_sentence
from rulebank.evidence import VALID_SPLIT, DraftSet
from rulebank.gate import base_edit
from rulebank.llm import Caller, FileCaller, json_schema
from rulebank.report import clean, delete_sentence, numbered, revise, rewrite_sentence, segment, sents


def instruction(drafts, k, rule, edit):
    """the rule's do text on draft k; edit = the edit as possible_edits names it, with its side or device type. Only the placeholders the text holds are filled"""
    text, f = rule["do"], rule["finding"]
    if P.SIDE in text:
        text = text.replace(P.SIDE, edit.split("->")[-1])
    if P.DEVICE_SENTENCE in text:
        text = text.replace(P.DEVICE_SENTENCE, DEVICE[edit.split(":")[-1]][0])
    if P.VALUE_SENTENCE in text:
        text = text.replace(P.VALUE_SENTENCE, value_sentence(drafts, k, f))
    if P.ASCENDING_MM in text:
        text = text.replace(P.ASCENDING_MM, str(round(float(drafts.ascending_mm[k]))))
    return text


def fired_rules(rules, ev, drafts, k):
    """[(rule id, rule, the edit as possible_edits names it)]: conditions hold and the draft is eligible (possible_edits lists the edit, also where the code edit abstains)"""
    v = drafts.ids[k]
    return fired_on(rules, ev, k, possible_edits(drafts, k, drafts.original[v], drafts.sentence_probs[v]))


def fired_on(rules, ev, k, eligible):
    """fired_rules with the draft's possible edits already listed"""
    out, added = [], set()
    for r in rules:
        if (r["edit"] in ADD_FAMILY and r["finding"] in added) or not rule_fires(r, ev, k):
            continue
        hit = [e for f, e, t in eligible if f == r["finding"] and base_edit(e) == r["edit"]]
        if hit:
            if r["edit"] in ADD_FAMILY:
                added.add(r["finding"])
            out.append((f"R{len(out) + 1}", r, hit[0]))
    return out


def evidence_sheet(fired, ev, k):
    lines = []
    for z in sorted({c[0] for _, r, _ in fired for c in r["when_code"]}):
        lines.append(f"- {P.evidence_line(z)}: {ev[z][k]:.4g}")
    return "\n".join(lines) or "(the rules below carry their own measurement)"


def schema(rule_ids, sentence_ids):
    edit = {"type": "object", "additionalProperties": False, "required": ["rule_id", "op", "sentence_id", "new_text"],
            "properties": {"rule_id": {"type": "string", "enum": rule_ids}, "op": {"type": "string", "enum": ["add", "rewrite", "remove"]},
                           "sentence_id": {"type": "integer", "enum": [0] + sentence_ids}, "new_text": {"type": "string"}}}
    return json_schema("edits", {"type": "object", "additionalProperties": False, "required": ["edits"], "properties": {"edits": {"type": "array", "items": edit}}})


def problem(answer, sentence_ids):
    """why an edit list cannot be applied, or None"""
    for e in answer["edits"]:
        if e["op"] == "add" and (e["sentence_id"] != 0 or not e["new_text"].strip()):
            return "add needs sentence_id 0 and a sentence"
        if e["op"] != "add" and e["sentence_id"] not in sentence_ids:
            return f"{e['op']} on sentence {e['sentence_id']}"
        if e["op"] == "rewrite" and not e["new_text"].strip():
            return "rewrite without a sentence"
    return None


def apply_edits(draft, answer, gen):
    """-> (revised text, new sentences dropped as repeats): a new sentence is written once, a rewrite that repeats it deletes its sentence, an addition that repeats it is dropped"""
    changes, adds, seen, repeats = [], [], {clean(p["text"]) for p in sents(draft)}, 0
    for e in answer["edits"]:
        new = clean(e["new_text"])
        if e["op"] == "remove":
            changes.append(delete_sentence(e["sentence_id"]))
        elif new in seen:
            repeats += 1
            if e["op"] == "rewrite":
                changes.append(delete_sentence(e["sentence_id"]))
        else:
            seen.add(new)
            changes.append(rewrite_sentence(e["sentence_id"], new)) if e["op"] == "rewrite" else adds.append(new)
    text = revise(draft, changes, adds)
    return (renumber(text) if gen == "nv" and changes else text), repeats


def ask(caller, key, prompt, sentence_ids, rule_ids, cfg, system):
    """the checked answer; None when every try failed"""
    for t in range(cfg.retries + 1):
        rec = caller.try_call(key if t == 0 else f"{key}|retry{t}", system, prompt, schema(rule_ids, sentence_ids), cfg.temperature if t == 0 else cfg.retry_temperature)
        if rec is not None and problem(rec["answer"], sentence_ids) is None:
            return rec["answer"]
    return None


def edit_drafts(gen, out, cfg, base_url, limit, exchange):
    """every bank of <out>/<gen>/banks.json on the validation drafts; limit > 0: only the first drafts with a rule that fired, for a smoke test. Exactly one of base_url (the local LLM) and exchange (rulebank.file_exchange: Claude)"""
    assert (base_url is None) != (exchange is None), "exactly one of base_url, exchange"
    d = os.path.join(out, gen)
    B = json.load(open(os.path.join(d, "banks.json")))
    drafts = DraftSet(gen, VALID_SPLIT)
    ev = drafts.evidence(sorted({c[0] for rules in B.values() for r in rules for c in r["when_code"]}))
    trace = os.path.join(d, "trace.jsonl")
    caller = FileCaller(exchange, trace, cfg) if exchange else Caller(base_url, trace, cfg)
    for name, rules in B.items():
        os.makedirs(os.path.join(d, name), exist_ok=False)
        rules = sorted(rules, key=lambda r: ORDER.index(r["edit"]))
        system = P.EDITOR_DRAFT_SYSTEM if any(c[0].startswith("draft") for r in rules for c in r["when_code"]) else P.EDITOR_SYSTEM
        jobs = []
        for k, v in enumerate(drafts.ids):
            fired = fired_rules(rules, ev, drafts, k)
            if fired:
                jobs.append((k, v, fired))
        jobs = jobs[:limit] if limit else jobs

        def one(job):
            k, v, fired = job
            pieces = segment(drafts.original[v])
            sid = [p["id"] for p in pieces if p["kind"] == "sent"]
            text = "\n".join(f"{i}: {instruction(drafts, k, r, e)}" for i, r, e in fired)
            return ask(caller, f"{gen}|{name}|{v}", P.editor_prompt(numbered(pieces), evidence_sheet(fired, ev, k), text), sid, [i for i, _, _ in fired], cfg, system)

        with ThreadPoolExecutor(cfg.workers) as ex:
            answers = list(ex.map(one, jobs))
        rows, n_fire, failed, n_rules, n_applied, n_repeats = [], collections.Counter(), 0, 0, 0, 0
        with open(os.path.join(d, name, "editor.jsonl"), "w") as f:
            for (k, v, fired), a in zip(jobs, answers):
                f.write(json.dumps({"vid": v, "rules": {i: f"{r['finding']}|{e}" for i, r, e in fired}, "answer": a}) + "\n")
                n_rules += len(fired)
                if a is None:
                    failed += 1
                    continue
                if not a["edits"]:                               # no edit: the draft stays as it is, render would tidy its spaces
                    continue
                used = {x["rule_id"] for x in a["edits"]}
                names = [f"{r['finding']}|{r['edit']}" for i, r, _ in fired if i in used]
                n_fire.update(names)
                n_applied += len(names)
                text, repeats = apply_edits(drafts.original[v], a, gen)
                n_repeats += repeats
                rows.append({"vid": v, "text": text, "fired": names})
        with open(os.path.join(d, name, "texts.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        json.dump({"changed": len(rows), "fired": dict(n_fire), "calls": len(jobs), "failed": failed, "rules_fired": n_rules, "rules_applied": n_applied, "repeated_sentences_dropped": n_repeats},
                  open(os.path.join(d, name, "fired.json"), "w"), indent=1)
        print(name, len(rules), "rules:", len(jobs), "calls,", failed, "failed,", len(rows), "changed,", n_applied, "of", n_rules, "fired rules applied", dict(n_fire), flush=True)
    if exchange:
        caller.finish()

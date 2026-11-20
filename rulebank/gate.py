"""The admission gate: always code, never the LLM. A rule is scored by lookup in the edit-outcome table on the development drafts it
may see, and enters the bank only if it passes; one rule per (abnormality, edit)."""
import re

import numpy as np

from rulebank.edits import aorta, liver
from rulebank.edits.spine import SCOLIO
from rulebank.labels import FINDINGS, NEGATED, macro_f1

DETAIL_EDITS = ("side", "add_value", "type", "add_typed")             # edits that carry their own evidence
# out-of-label abnormalities: the reference text is the only check; a removal needs the references to disagree, an addition to agree
OUT_OF_LABEL = {"aortic dilation": aorta.stated, "hepatic steatosis": liver.stated,
           "scoliosis": lambda r: any(SCOLIO.search(s) and not NEGATED.search(s) for s in re.split(r"(?<=[.;])\s+", r))}


def base_edit(e):
    return e.split(":")[0].split("->")[0]


def min_changed(V, n_all, g):
    return max(1, int(round(g.min_changed_per_3000 * len(V) / n_all)))


def rule_outcome(rule, V, table, drafts, base_f1):
    """rule's outcome on the drafts V (a set of ids): n changed, macro F1 change over V, mean paired GREEN change"""
    ks = [k for k in rule["rows"] if table.rows[k]["vid"] in V]
    if not ks:
        return None
    vi = sorted(V, key=lambda v: table.position[v])
    pos = {v: n for n, v in enumerate(vi)}
    P = drafts.labels[[table.position[v] for v in vi]].copy()
    for k in ks:
        P[pos[table.rows[k]["vid"]]] = table.labels[k]
    f1 = macro_f1(P, drafts.reference_labels[[table.position[v] for v in vi]]) - base_f1
    g = table.dgreen[ks]
    s = {"n": len(ks), "dF1": float(f1), "dG": float(np.nanmean(g)) if np.isfinite(g).any() else None}
    if rule["finding"] in FINDINGS:
        j = FINDINGS.index(rule["finding"])
        s["precision"] = float(drafts.reference_labels[[table.position[table.rows[k]["vid"]] for k in ks], j].mean())
    if rule["finding"] in OUT_OF_LABEL:
        s["agree"] = float(np.mean([OUT_OF_LABEL[rule["finding"]](drafts.references[table.rows[k]["vid"]]) for k in ks]))
    return s


def refused(rule, s, min_n, g):
    """why the gate refuses a rule, None when it passes: macro F1 rises and the GREEN change is at least the margin (margin_add for a labelled addition), a labelled
    addition reaches the precision floor;
    detail edits: GREEN only; out-of-label: GREEN and agreement with the references"""
    if s is None:
        return "changes no draft"
    if s["n"] < min_n:
        return "changes too few drafts"
    if s["dG"] is None:
        return "no GREEN score"
    f, e = rule["finding"], rule["edit"]
    labelled_add = e == "add" and f in FINDINGS
    if s["dG"] < (g.margin_add if labelled_add else g.margin):
        return "GREEN fell"
    if e in DETAIL_EDITS:
        return None
    if f in OUT_OF_LABEL:
        agrees = (s["agree"] <= g.max_agree_remove) if e != "add" else (s["agree"] >= g.min_agree_add)
        return None if agrees else "the references do not support it"
    if s["dF1"] <= 0:
        return "macro F1 did not rise"
    if e == "add" and s["precision"] < g.min_precision_add:
        return "precision below the floor"
    return None


def rank_value(s, rule):
    """what the gate maximises among the admitted rules of one (abnormality, edit)"""
    return s["n"] if (rule["finding"] in OUT_OF_LABEL or rule["edit"] in DETAIL_EDITS) else s["dF1"]


def select_bank(cands, V, table, drafts, g):
    """the bank: for each (abnormality, edit) the admitted candidate that ranks highest -> list of rule ids"""
    vh = [table.position[v] for v in V]
    base_f1 = macro_f1(drafts.labels[vh], drafts.reference_labels[vh])
    min_n = min_changed(V, len(drafts.ids), g)
    best = {}
    for r in cands:
        s = rule_outcome(r, V, table, drafts, base_f1)
        if refused(r, s, min_n, g) is None:
            k = (r["finding"], r["edit"])
            if k not in best or rank_value(s, r) > rank_value(best[k][1], best[k][0]):
                best[k] = (r, s)
    return [r["id"] for r, s in best.values()]

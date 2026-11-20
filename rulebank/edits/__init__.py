"""The edit library: for one draft, every edit the grammar allows and the text it produces. A rule only decides which drafts receive
an edit; the edit itself is code, and it abstains (None) when it cannot change the report cleanly."""
import re

import numpy as np

from rulebank.edits import aorta, liver
from rulebank.edits.add import ADD_SENT, EXTRA_OTHER, OTHER_WORD, TARGET_TERM, add_abnormality
from rulebank.edits.details import detail_edits
from rulebank.edits.heart import CM, CO, coronary_cut, heart_cut
from rulebank.edits.spine import SCOLIO, scoliosis_cut
from rulebank.labels import DENIAL, FINDINGS, KEY, NEGATED
from rulebank.report import delete_sentence, revise, segment, sents

NORMAL_IMPRESSION = "1. No acute abnormality in the chest."


def renumber(text):
    """NV's 'IMPRESSION:' list after a removal: items renumbered 1..n, empty lines dropped; empty -> NV's normal impression"""
    i = text.find("IMPRESSION:")
    if i < 0:
        return text
    head, body = text[:i + len("IMPRESSION:")], text[i + len("IMPRESSION:"):]
    items = [re.sub(r"^\s*\d+\.\s*", "", l).strip() for l in body.split("\n")]
    items = [x for x in items if x and not re.fullmatch(r"\d+\.?", x)]
    return head + "\n\n" + ("\n".join(f"{n}. {x}" for n, x in enumerate(items, 1)) if items else NORMAL_IMPRESSION)


def remove_statements(text, probs, j):
    """delete the sentences that state only abnormality j; a mixed sentence makes the edit abstain. probs = {sentence id of text: RadBERT}"""
    lab = FINDINGS[j]
    term = TARGET_TERM.get(lab, KEY[j].pattern)
    by_text = {p["text"]: probs[p["id"]] for p in sents(text)}
    bad = {s for s, pr in by_text.items() if pr[j] >= 0.5 and not DENIAL.search(s)}
    ch = []
    for p in segment(text):
        if p["kind"] != "sent" or p["text"] not in bad:
            continue
        others = np.delete(np.asarray(by_text[p["text"]]), j)
        rest = re.sub(term, "", p["text"], flags=re.I)
        if (others >= 0.5).any() or OTHER_WORD.search(rest) or EXTRA_OTHER.search(rest):
            return None
        ch.append(delete_sentence(p["id"]))
    return revise(text, ch, []) if ch else None


def remove_label(text, probs, j):
    if j in (CM, CO):
        bad = {p["text"] for p in sents(text) if probs[p["id"]][j] >= 0.5}
        if j == CM:
            return heart_cut(text, bad)
        cut = [coronary_cut(p) for p in sents(text) if p["text"] in bad]
        return revise(text, cut, []) if cut and all(cut) else None
    return remove_statements(text, probs, j)


def possible_edits(drafts, k, d, probs):
    """[(finding, edit, text or None)] for case k of the draft set. d = the text to edit (the draft, or the draft after earlier rules),
    probs = {sentence id of d: per-sentence RadBERT}"""
    out = []
    for j, lab in enumerate(FINDINGS):
        if not drafts.labels[k, j]:
            out.append((lab, "add", add_abnormality(d, ADD_SENT[lab], j, drafts.gen)))
        else:
            out.append((lab, "remove", remove_label(d, probs, j)))
    if liver.stated(d):
        out.append(("hepatic steatosis", "remove", liver.remove_safe(d)))
    else:
        out.append(("hepatic steatosis", "add", liver.add(d)))
    if aorta.stated(d):
        out.append(("aortic dilation", "rewrite_normal", aorta.rewrite_normal(d)))
    elif drafts.ascending_mm[k] == drafts.ascending_mm[k] and not any(aorta.dil_clause(p["text"]) for p in sents(d)):
        out.append(("aortic dilation", "add", revise(d, [], [f"The ascending aorta is dilated, measuring {round(float(drafts.ascending_mm[k]))} mm."])))
    sc = [p for p in sents(d) if SCOLIO.search(p["text"]) and not NEGATED.search(p["text"])]
    if sc:
        cut = [scoliosis_cut(p) for p in sc]
        out.append(("scoliosis", "remove", revise(d, cut, []) if all(cut) else None))
    out += detail_edits(drafts, k, d, probs)
    if drafts.gen == "nv":
        out = [(f, e, renumber(t) if (t is not None and (e in ("remove", "rewrite_normal") or e.startswith("side->"))) else t) for f, e, t in out]
    return [(f, e, (None if t is None or t == d else t)) for f, e, t in out]

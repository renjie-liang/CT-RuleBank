"""Detail edits: effusion side, measured diameters, device type."""
import re

import numpy as np

from rulebank.edits import aorta
from rulebank.edits.add import EXTRA_OTHER, OTHER_WORD, add_abnormality
from rulebank.labels import FINDINGS, NEGATED
from rulebank.report import delete_sentence, revise, rewrite_sentence, segment, sents

SIDE_SENT = {"bilateral": "Bilateral pleural effusion is observed.", "right": "Pleural effusion is observed on the right.",
             "left": "Pleural effusion is observed on the left."}
DEVICE = {"sternotomy wires": ("Sternotomy wires are observed.", r"sternotom|sternal (wire|suture)"),
          "pacemaker / ICD": ("A cardiac pacemaker is observed.", r"pacemaker|\bICD\b|defibrillator|\blead"),
          "valve prosthesis": ("A prosthetic heart valve is observed.", r"valve"),
          "spinal fixation": ("Spinal fixation material is observed.", r"fixation|screw|\brod")}
MM = re.compile(r"\d+(\.\d+)?\s*(mm|cm)\b", re.I)
VALUE_SITES = {"ascending aorta": (r"aort", "ascending_mm", "The ascending aorta measures {} mm."), "aortic arch": (r"aort", "arch_mm", "The aortic arch measures {} mm."),
               "descending aorta": (r"aort", "descending_mm", "The descending aorta measures {} mm."),
               "main pulmonary artery": (r"pulmonary arter", "pa_mm", "The main pulmonary artery measures {} mm.")}
EFF_SENT = re.compile(r"([^.]*pleural effusion[^.]*)\.", re.I)
EFF_NEG = re.compile(r"\bno\b|\bnot\b|without|absent|free of", re.I)
SIZE = re.compile(r"\b(minimal|trace|mild|small|moderate|large|massive|significant|extensive)\b", re.I)


def text_side(text):
    """side words over the positive effusion sentences of a text -> right / left / bilateral / unspecified / none"""
    sides, pos = set(), 0
    for s in EFF_SENT.findall(text):
        if EFF_NEG.search(s):
            continue
        pos += 1
        if re.search(r"\b(bilateral|both)\b", s, re.I):
            sides.add("bilateral")
        else:
            for w in ("right", "left"):
                if re.search(rf"\b{w}\b", s, re.I):
                    sides.add(w)
    if pos == 0:
        return "none"
    if "bilateral" in sides or sides == {"right", "left"}:
        return "bilateral"
    if len(sides) == 1:
        return sides.pop()
    return "unspecified"


def side_edit(d, probs, ss):
    """the draft's effusion sentences rewritten to one sentence with the measured side (a size word is kept), or None on a mixed sentence"""
    PE = FINDINGS.index("Pleural effusion")
    ch = []
    for p in segment(d):
        if p["kind"] == "sent" and re.search(r"pleural effusion", p["text"], re.I) and not NEGATED.search(p["text"]):
            others = np.delete(np.asarray(probs[p["id"]]), PE)
            rest = re.sub(r"pleural effusion|bilateral|both|right|left|hemithorax|side", "", p["text"], flags=re.I)
            if (others >= 0.5).any() or OTHER_WORD.search(rest) or EXTRA_OTHER.search(rest):
                return None
            if not ch:
                m = SIZE.search(p["text"])
                new = SIDE_SENT[ss] if m is None else (f"Bilateral {m.group(1).lower()} pleural effusion is observed." if ss == "bilateral"
                                                      else f"{m.group(1).capitalize()} pleural effusion is observed on the {ss}.")
                ch.append(rewrite_sentence(p["id"], new))
            else:
                ch.append(delete_sentence(p["id"]))
    return revise(d, ch, []) if ch else None


def value_sentence(drafts, k, site):
    """the sentence with the measured diameter, or None without a measurement"""
    word, column, sentence = VALUE_SITES[site]
    x = drafts.measurements[column].iloc[k]
    return sentence.format(round(float(x))) if x == x else None


def detail_edits(drafts, k, d, probs):
    """[(finding, edit, text or None)] of the detail edits for case k; d and probs as in possible_edits"""
    out = []
    ds, ss = text_side(d), drafts.side[k]
    if ds in SIDE_SENT and ss in SIDE_SENT and ds != ss:
        out.append(("Pleural effusion", f"side->{ss}", side_edit(d, probs, ss)))
    for site, (word, _, _) in VALUE_SITES.items():
        named = [p["text"] for p in sents(d) if re.search(word, p["text"], re.I)]
        new = value_sentence(drafts, k, site)
        if new is not None and named and not any(MM.search(x) for x in named):
            out.append((site, "add_value", revise(d, [], [new])))
    MMI = FINDINGS.index("Medical material")
    for ty in sorted(t for t in drafts.device_types[k] if t in DEVICE):
        sent, rx = DEVICE[ty]
        if re.search(rx, d, re.I):
            continue
        if drafts.labels[k, MMI]:
            out.append(("Medical material", f"type:{ty}", revise(d, [], [sent])))
        else:
            out.append(("Medical material", f"add_typed:{ty}", add_abnormality(d, sent, MMI, drafts.gen)))
    return out

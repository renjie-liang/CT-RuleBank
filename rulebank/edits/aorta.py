"""Aortic dilation in report text: does a report state it, cut it, rewrite the report to a normal aorta."""
import re

from rulebank.labels import DENIAL
from rulebank.report import delete_sentence, revise, rewrite_sentence, sents

AO = r"(ascending|descending|thoracic|abdominal)?\s*aorta\b|aortic (root|arch|bulb)"
OTHER = re.compile(r"calcif|plaque|atherom|nodul|effusion|lymph|emphysem|opacit|consolidat|atelecta|cardiomeg|heart|thicken|hernia|bronch|mass", re.I)
WORD = r"dilat|ectat|ectasi|aneurysm|enlarg|widen|wider than normal|larger than normal|above normal|increased in (diameter|calib)"
DIL = re.compile(rf"({AO})[^.]*({WORD})|({WORD})[^.]*({AO})", re.I)
NEG = re.compile(r"\bno\b|\bnot\b|(?<!than )(?<!above )\bnormal\b|within normal", re.I)
PREFIX = re.compile(r"^((?:Findings|Impression)\s*:\s*)?(\d+\.\s*)?", re.I)
ITEM_OTHER = re.compile(OTHER.pattern + r"|pulmonary|trunk|valv|stenos|thromb|dissect", re.I)
NORMAL_SENT = "Thoracic aorta diameter is normal."
HAS_NORMAL = re.compile(rf"({AO})[^.]*(diamet|calib|width|caliber)[^.]*(normal|natural)|(diamet|calib|width|caliber)[^.]*({AO})[^.]*(normal|natural)", re.I)
ARCH_DESC = re.compile(r"^(?!.*ascending)(?=.*\barch\b)(?=.*descending)[^.]*(normal|natural)[^.]*\.?$", re.I)
WHOLE_NUM = re.compile(r"\d")
# clause-level test used before adding: "wider than normal" is a positive statement
AORTA = r"\baort(?:a|ic)\b"                       # not "aortopulmonary" / "aorticopulmonary" (lymph node stations)
CLAUSE_DIL = re.compile(rf"^(?!.*lymph).*(?:{AORTA}[^.]*(dilat|ectat|aneurysm|enlarg|wide)|(dilat|ectat|aneurysm|enlarg|wide)\w*[^.]*{AORTA})", re.I)
THAN_NORMAL = re.compile(r"\b(wider|larger|greater|bigger|more)\s+than\s+normal\b", re.I)


def clause_states(t):
    return CLAUSE_DIL.search(t) is not None and DENIAL.search(THAN_NORMAL.sub("", t)) is None


def dil_clause(t):
    """aortic dilation stated in one comma / semicolon clause of the sentence"""
    return any(clause_states(c) for c in re.split(r"[,;]", t))


def is_dilation(text):
    return bool(DIL.search(text)) and not NEG.search(text) and not re.search("lymph", text, re.I)


def items(text):
    return [re.sub(r"^and\s+", "", x.strip()) for x in re.split(r",\s*", text)]


def mentions(text):
    """a sentence states aortic dilation if one of its comma-separated items does: a sentence-level match runs across items
    ('dilatation in the pulmonary artery, calcific atheroma plaques in the aorta' read as aortic dilation)"""
    return any(is_dilation(x) for x in items(text))


def stated(t):
    return any(mentions(p["text"]) for p in sents(t))


def cut_sentence(text):
    """-> ('remove', '') | ('rewrite', new text) | None (no clean cut)"""
    if not ITEM_OTHER.search(text):         # names no other finding: the whole sentence is about the dilation
        return "remove", ""
    pre = PREFIX.match(text).group(0)
    body = text[len(pre):].strip().rstrip(".").strip()
    parts = [x.strip() for x in re.split(r",\s*", body)]
    if len(parts) < 2:
        return None
    keep, drop = [], []
    for x in parts:
        core = re.sub(r"^and\s+", "", x)
        if is_dilation(core):
            if ITEM_OTHER.search(core):
                return None
            drop.append(x)
        else:
            keep.append(x)
    if not drop or not keep:
        return None
    keep[0] = re.sub(r"^and\s+", "", keep[0])
    keep[0] = keep[0][0].upper() + keep[0][1:]
    new = pre + ", ".join(keep) + "."
    return None if mentions(new) else ("rewrite", new)


def rewrite_normal(t):
    """the report with its dilation statements replaced by a normal aorta, or None when a sentence cannot be cut cleanly"""
    ch, rewritten = [], False
    pre = [p for p in sents(t) if not mentions(p["text"]) and ARCH_DESC.search(p["text"]) and not WHOLE_NUM.search(p["text"])]
    if pre:
        ch.append(rewrite_sentence(pre[0]["id"], NORMAL_SENT))
        rewritten = True
    has_normal = any(HAS_NORMAL.search(p["text"]) and not mentions(p["text"]) and not ARCH_DESC.search(p["text"]) for p in sents(t))
    for p in sents(t):
        if not mentions(p["text"]):
            continue
        r = cut_sentence(p["text"])
        if r is None:
            return None
        if r[0] == "remove" and not rewritten and not has_normal:
            ch.append(rewrite_sentence(p["id"], NORMAL_SENT))
            rewritten = True
        elif r[0] == "remove":
            ch.append(delete_sentence(p["id"]))
        else:
            ch.append(rewrite_sentence(p["id"], r[1]))
    if len(ch) == len(pre[:1]):          # no dilation sentence handled (only the arch / descending rewrite)
        return None
    new = revise(t, ch, [NORMAL_SENT] if not rewritten and not has_normal else [])
    assert not stated(new), new
    return new

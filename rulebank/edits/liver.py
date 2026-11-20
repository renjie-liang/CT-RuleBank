"""Hepatic steatosis in report text."""
import re

from rulebank.report import delete_sentence, revise, rewrite_sentence, sents

STEA = re.compile(r"steatos|fatty liver|fatty infiltration|hepatosteatosis|adiposity|(liver|hepatic)[^.]*(hypodens|density (is |was |has )?(decreas|reduc|low))"
                  r"|(decreas|reduc)[^.]*density[^.]*(liver|hepat|steato)", re.I)
NEG = re.compile(r"\bno\b|\bnot\b|without|\bnormal\b", re.I)
ITEM_OTHER = re.compile(r"megal|spleen|splen|kidney|renal|gallbladder|cyst|lesion|nodul|mass|calcif|effusion|lymph|hernia|cardiomeg|aort|lung|pleur", re.I)
PREFIX = re.compile(r"^((?:Findings|Impression)\s*:\s*)?(\d+\.\s*)?", re.I)
SENT = "Hepatosteatosis."


def is_stea(text):
    return bool(STEA.search(text)) and not NEG.search(text)


def items(text):
    return [re.sub(r"^and\s+", "", x.strip()) for x in re.split(r",\s*", text)]


def mentions(text):
    return any(is_stea(x) for x in items(text))


def stated(t):
    return any(mentions(p["text"]) for p in sents(t))


def cut_sentence(text):
    """-> ('remove', '') | ('rewrite', new text) | None (no clean cut)"""
    if not ITEM_OTHER.search(text):
        return "remove", ""
    pre = PREFIX.match(text).group(0)
    parts = [x.strip() for x in re.split(r",\s*", text[len(pre):].strip().rstrip(".").strip())]
    keep, drop = [], []
    for x in parts:
        core = re.sub(r"^and\s+", "", x)
        if is_stea(core):
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


def remove_safe(t):
    ch = []
    for p in sents(t):
        if not mentions(p["text"]):
            continue
        r = cut_sentence(p["text"])
        if r is None:
            return None
        ch.append(delete_sentence(p["id"]) if r[0] == "remove" else rewrite_sentence(p["id"], r[1]))
    new = revise(t, ch, [])
    assert not stated(new), new
    return new


def add(t):
    new = revise(t, [], [SENT])
    assert stated(new), new
    return new

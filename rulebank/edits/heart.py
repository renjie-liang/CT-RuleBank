"""Cardiomegaly and coronary calcification: removing them from a report."""
import re

from rulebank.labels import FINDINGS
from rulebank.report import delete_sentence, revise, rewrite_sentence, segment, sents

CM, CO = FINDINGS.index("Cardiomegaly"), FINDINGS.index("Coronary artery wall calcification")
HEART_NORMAL = "Heart contour and size are normal."
HEART_ITEM = re.compile(r"cardiomeg|heart|cardiothoracic|\bcto\b", re.I)
OTHER_FINDING = re.compile(r"aort|main vascular|vascular structures|dilat|ectat|aneurysm|nodul|effusion|lymph|emphysem|opacit|consolidat|atelecta|cardiomeg|heart size|thicken|hernia|bronch|valv", re.I)
CORONARY_CUT = [re.compile(r",?\s+and\s+(the\s+)?((left|right)\s+(main\s+)?)?coronary\s+arter(y|ies)", re.I),
                re.compile(r"(the\s+)?coronary\s+arter(y|ies)\s+and\s+", re.I)]
CORONARY_WORD = re.compile(r"coronar", re.I)


def coronary_cut(p):
    """the safe edit of one sentence that states coronary calcification, or None"""
    t = p["text"]
    if not OTHER_FINDING.search(CORONARY_WORD.sub("", t)):
        return delete_sentence(p["id"])
    new = t
    for rx in CORONARY_CUT:
        new = rx.sub("", new, count=1)
    return rewrite_sentence(p["id"], new) if new != t and not CORONARY_WORD.search(new) else None


def heart_cut(t, bad):
    """remove cardiomegaly: list sentences lose only their heart / cardiomegaly / CTO item; heart-only sentences: the first is rewritten
    to HEART_NORMAL unless the report already has a normal heart statement, the rest deleted; None if a sentence cannot be cut cleanly"""
    ch, rewrote = [], any(re.search(r"heart (contour|size)[^.]*normal|normal heart size", p["text"], re.I) for p in sents(t))
    # only sentences that name the heart: the draft's per-sentence RadBERT also reads 'Hepatosplenomegaly.' as cardiomegaly
    for p in [p for p in segment(t) if p["kind"] == "sent" and p["text"] in bad and HEART_ITEM.search(p["text"])]:
        body = re.sub(r"^((?:Impression|Findings)\s*:\s*)", "", p["text"]).strip().rstrip(".")
        pre = p["text"][:len(p["text"]) - len(p["text"].lstrip())] + p["text"].strip()[:len(p["text"].strip()) - len(re.sub(r"^((?:Impression|Findings)\s*:\s*)", "", p["text"].strip()))]
        parts = [x.strip() for x in re.split(r",\s*", body) if x.strip()]
        keep = [x for x in parts if not HEART_ITEM.search(x)]
        if len(parts) > 1 and keep and len(keep) < len(parts):
            keep[0] = keep[0][0].upper() + keep[0][1:]
            ch.append(rewrite_sentence(p["id"], pre + ", ".join(keep) + "."))
        elif len(parts) > 1:
            return None
        elif not rewrote:
            ch.append(rewrite_sentence(p["id"], HEART_NORMAL))
            rewrote = True
        else:
            ch.append(delete_sentence(p["id"]))
    return revise(t, ch, []) if ch else None

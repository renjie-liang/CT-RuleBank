"""Scoliosis in report text."""
import re

from rulebank.report import delete_sentence, rewrite_sentence

SCOLIO = re.compile(r"scolio", re.I)
SPINE_OTHER = re.compile(r"osteopor|osteopen|kyphos|degenerat|spur|osteophyt|height|fracture|bridging|lytic|sclerot|hemangiom|lordos", re.I)


def scoliosis_cut(p):
    """safe edit of one sentence that states scoliosis: a scoliosis-only sentence is deleted; in a list ('Osteoporosis, thoracic
    scoliosis, increased thoracic kyphosis, and ...') only the scoliosis item is cut; None when it cannot be cut cleanly"""
    t = p["text"]
    if not SPINE_OTHER.search(t):
        return delete_sentence(p["id"])
    pre = re.match(r"^((?:\d+\.\s*)?)", t).group(1)
    body = t[len(pre):].strip().rstrip(".")
    parts = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", body) if x.strip()]
    keep = [x for x in parts if not SCOLIO.search(x)]
    if len(keep) == len(parts) or not keep or any(SCOLIO.search(x) and SPINE_OTHER.search(x) for x in parts):
        return None
    keep[0] = keep[0][0].upper() + keep[0][1:]
    return rewrite_sentence(p["id"], pre + (", ".join(keep[:-1]) + " and " + keep[-1] if len(keep) > 1 else keep[0]) + ".")

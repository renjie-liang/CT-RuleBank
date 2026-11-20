"""Adding an abnormality to a report: the sentence, and what happens to a draft sentence that denies it."""
import re

from rulebank.labels import DENIAL, FINDINGS, KEY
from rulebank.report import delete_sentence, revise, rewrite_sentence, sents

CARDIO_SENT = "The heart size is increased, consistent with cardiomegaly."
HIATAL_SENT = "Sliding type hiatal hernia was observed at the lower end of the esophagus."
KEEP_VESSELS = "Mediastinal main vascular structures are normal."
DENY = {"selfcue": {"Cardiomegaly": re.compile(r"heart contour|heart size|cardiothoracic", re.I)}}
POSITIVE = {"Medical material": "A medical device is observed.",
            "Arterial wall calcification": "Atheroma plaques are observed in the aorta.",
            "Cardiomegaly": "The heart is larger than normal.",
            "Pericardial effusion": "Pericardial effusion is observed.",
            "Coronary artery wall calcification": "Calcific atheroma plaques are observed in the coronary arteries.",
            "Hiatal hernia": "There is a hiatal hernia.",
            "Lymphadenopathy": "There are pathologically enlarged lymph nodes in the mediastinum.",
            "Emphysema": "There are emphysematous changes in both lungs.",
            "Atelectasis": "There are linear atelectasis in both lungs.",
            "Lung nodule": "There are millimetric nonspecific nodules in both lungs.",
            "Lung opacity": "There are ground glass densities in both lungs.",
            "Pulmonary fibrotic sequela": "There are pleuroparenchymal sequelae changes in both lungs.",
            "Pleural effusion": "Minimal pleural effusion is observed.",
            "Mosaic attenuation pattern": "A mosaic attenuation pattern was observed in both lungs.",
            "Peribronchial thickening": "Peribronchial thickenings are observed in both lungs.",
            "Consolidation": "There are consolidation areas in both lungs.",
            "Bronchiectasis": "There are bronchiectasis in both lungs.",
            "Interlobular septal thickening": "Interlobular septal thickenings are observed in both lungs."}
LYMPH_SENT = "Enlarged lymph nodes are observed in the mediastinum."
PERICARDIAL_SENT = "Minimal pericardial effusion is observed."
# the CT-RATE wording of each abnormality; these four were chosen by how RadBERT reads them in a report (docs/edit_library_next.md)
ADD_SENT = {**POSITIVE, "Cardiomegaly": CARDIO_SENT, "Hiatal hernia": HIATAL_SENT, "Lymphadenopathy": LYMPH_SENT, "Pericardial effusion": PERICARDIAL_SENT}
OTHER_WORD = re.compile(r"\b(nodul\w*|mass\w*|infiltrat\w*|consolidat\w*|atelecta\w*|effusion|thicken\w*|lymph\w*|lesion\w*|opacit\w*|emphysem\w*|hernia|"
                        r"bronchiect\w*|cardiomeg\w*|pericard\w*|pleura\w*|calcif\w*|plaque\w*|mosaic|septal|densit\w*)\b", re.I)
EXTRA_OTHER = re.compile(r"ground.?glass|reticul\w*|septal|interlobular|bronch\w*|airway|fibro\w*|scar\w*|cyst\w*|aeration|hernia|"
                         r"aneurysm\w*|dilat\w*|ectas\w*|fracture\w*|osteo\w*|spondyl\w*|stent|graft|catheter|wire|clip\w*|valve\w*|"
                         r"pneumo\w*|nodul\w*|mass\w*|lesion\w*|thicken\w*|calcif\w*|plaque\w*|atherom\w*|emphysem\w*|lymph\w*", re.I)
TARGET_TERM = {"Lung nodule": r"nodul\w*|nodule\w*", "Atelectasis": r"atelecta\w*", "Consolidation": r"consolidat\w*", "Hiatal hernia": r"hiatal hernia|hiatus hernia|hernia",
               "Coronary artery wall calcification": r"coronary\w*(\s+arter(y|ies))?", "Cardiomegaly": r"cardiomegal\w*", "Lung opacity": r"opacit\w*|ground.glass",
               "Pericardial effusion": r"pericardial effusion", "Pulmonary fibrotic sequela": r"sequel\w*"}


def cut_term(text, term):
    """drop the target term and its joining word from a denial list ('no nodular or infiltrative lesion' -> 'no infiltrative lesion',
    'No mass, nodule, or infiltration' -> 'No mass or infiltration'); None if the term remains or nothing else is denied"""
    rx = re.compile(rf"(\s*,\s*(or|and)?\s*|\s+(or|and)\s+)?\b({term})\b(\s*,\s*|\s+(or|and)\s+|\s+)?", re.I)
    m = rx.search(text)
    if m is None:
        return None
    before, after = (m.group(1) or ""), (m.group(5) or "")
    joiner = " " if before and after else (" " if after or before else "")
    if before and after and re.search(r"\b(or|and)\b", before + after, re.I):
        joiner = " or " if re.search(r"\bor\b", before + after, re.I) else " and "
    new = (text[:m.start()] + joiner + text[m.end():]).replace("  ", " ")
    new = re.sub(r"\s+([,.])", r"\1", re.sub(r",\s*,", ",", new)).strip()
    if re.search(rf"\b({term})\b", new, re.I) or not OTHER_WORD.search(new):
        return None
    return new[0].upper() + new[1:]


def denials(draft, j, gen):
    """the draft sentences that name abnormality j in negated form"""
    deny = DENY.get(gen, {}).get(FINDINGS[j])
    return [p for p in sents(draft) if (KEY[j].search(p["text"]) or (deny is not None and deny.search(p["text"]))) and DENIAL.search(p["text"])]


def add_sentence(draft, sentence, j, gen):
    """RadBERT reads a denial that follows an appended positive as negative, so the first denial of the abnormality is REWRITTEN to the
    positive sentence and further denials are removed; with no denial the sentence is appended to the findings"""
    neg = denials(draft, j, gen)
    if not neg:
        return revise(draft, [], [sentence])
    first = (KEEP_VESSELS + " " + sentence) if re.search(r"vascular", neg[0]["text"], re.I) else sentence
    return revise(draft, [rewrite_sentence(neg[0]["id"], first)] + [delete_sentence(p["id"]) for p in neg[1:]], [])


def add_abnormality(draft, sentence, j, gen):
    """add_sentence, except that a denial naming other findings too is not rewritten whole: the target term is cut from it and the
    positive sentence appended; None when the cut is not clean (the edit abstains)"""
    lab = FINDINGS[j]
    neg = denials(draft, j, gen)
    mixed = [p for p in neg if OTHER_WORD.search(re.sub(TARGET_TERM.get(lab, KEY[j].pattern), "", p["text"], flags=re.I))]
    if not mixed:
        return add_sentence(draft, sentence, j, gen)
    ch = []
    for p in neg:
        if p in mixed:
            new = cut_term(p["text"], TARGET_TERM.get(lab, KEY[j].pattern))
            if new is None:
                return None
            ch.append(rewrite_sentence(p["id"], new))
        else:
            ch.append(delete_sentence(p["id"]))
    return revise(draft, ch, [sentence])

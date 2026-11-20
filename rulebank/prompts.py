"""Everything the rule agent reads: the system prompt, the grammar, the initial bank, the experience kept from earlier rounds and the round instruction.
Each round is one LLM call: system_prompt(cfg) and user_prompt(...). The evaluators' own prompts are not here: GREEN in eval/green_score/, CT-FineBench in finebench/prompts.py."""
import collections

from rulebank.edits.add import ADD_SENT
from rulebank.edits.liver import SENT as LIVER_SENT

THRESHOLDS = [f"p{p:g}" for p in [5, 10, 20, 30, 50, 70, 80, 90, 95, 97.5, 99]]    # the 11 thresholds code tries for every probe

# ---- system prompt ----
EXPLORE = (
    "You learn a rule bank that corrects automatically generated chest CT report drafts of one report generator. A rule decides "
    "which drafts receive one predefined edit for one finding, based on one piece of independent evidence from the same scan. The bank "
    "starts with the EVIDENCE list only: what each value measures, nothing about which finding it bears on, in which direction, or at "
    "which threshold. You find that out by exploring. Each round you propose probes: a finding, an edit, an evidence and a direction "
    "(>= or <=). You never choose a threshold: code tries 11 percentile thresholds of the evidence on the fit drafts and reports the best "
    "one: drafts changed, macro F1 change over the 18 CT-RATE labels, and the paired GREEN change on the changed reports. The gate (code) "
    "admits a threshold when macro F1 rises and the GREEN change is at least zero (out-of-label findings: GREEN >= 0 and agreement with "
    "the references), and keeps the admitted rule with the largest macro F1 gain per finding and edit. Your outcomes so far are kept in "
    "the EXPERIENCE section. Use your medical knowledge to decide what to try, and the outcomes to decide what to try next; do not repeat "
    "a probe: a finding, edit, evidence and direction marked as swept has been tried at every threshold. Answer with JSON only.")
EXPLORE_P2 = (
    "Goal: build a set of correction rules for the chest CT report drafts of one report model. Each rule says which drafts get one "
    "predefined edit for one finding, judged from one piece of independent evidence computed on the same scan. At the start the bank "
    "holds the EVIDENCE list only: what each value measures, nothing about which finding it bears on, in which direction, or at which "
    "threshold. Discover that yourself. In every round, submit probes: finding, edit, evidence, direction (>= or <=). Thresholds are not "
    "yours to choose: code tries 11 percentile thresholds on the fit drafts and reports the best: how many drafts changed, the change in "
    "macro F1 over the 18 CT-RATE labels, and the paired GREEN change on those drafts. A code gate accepts a threshold if macro F1 goes up "
    "and the GREEN change is at least zero (out-of-label findings: GREEN >= 0 plus agreement with the references); for each finding and "
    "edit it keeps the accepted rule with the highest macro F1 gain. Past outcomes appear in the EXPERIENCE section. Decide what to try from "
    "medical knowledge and from those outcomes, and never repeat a probe: anything marked as swept was tried at every threshold. Answer with JSON only.")
SYSTEM = {"explore": EXPLORE, "explore_p2": EXPLORE_P2}
ANON_OLD = "the EVIDENCE list only: what each value measures, nothing about which finding it bears on, in which direction, or at which threshold."
ANON_NEW = "the EVIDENCE list only: anonymous identifiers, nothing about what a value measures, which finding it bears on, in which direction, or at which threshold."
NO_EXPERIENCE_OLD = "Your outcomes so far are kept in the EXPERIENCE section."
BLIND = [("and reports the best one: drafts changed, macro F1 change over the 18 CT-RATE labels, and the paired GREEN change on the changed reports.", "and the gate scores each; you are not told any score."),
         ("Your outcomes so far are kept in the EXPERIENCE section.", "You do not see any outcome: the EXPERIENCE section lists only the probes tried so far."),
         ("Use your medical knowledge to decide what to try, and the outcomes to decide what to try next;", "Use your medical knowledge to decide what to try;")]
DRAFT_ONLY_OLD = "one piece of independent evidence from the same scan"
DRAFT_ONLY_NEW = "one signal computed from the text of the draft itself, with no information from the image"
NO_EXPERIENCE_NEW = "You see only the outcomes of the last round."


def system_prompt(cfg):
    system = SYSTEM[cfg.learner.prompt]
    if cfg.gate.margin_add != cfg.gate.margin:                       # a labelled addition has its own GREEN margin (RJ 2026-11-08); the text is unchanged when the two are equal
        assert system.count("at least zero (out-of-label findings:") == 1
        system = system.replace("at least zero (out-of-label findings:", f"at least zero (the edit add on one of the 18 labels: at least {cfg.gate.margin_add}; out-of-label findings:")
    if cfg.gate.min_precision_add:                                   # the precision floor of a labelled addition (RJ 2026-11-08)
        assert system.count("out-of-label findings:") == 1
        system = system.replace("out-of-label findings:", f"an add on one of the 18 labels must also reach a precision of {100 * cfg.gate.min_precision_add:.0f}% against the reference labels; out-of-label findings:")
    if cfg.evidence == "draft_only":                                 # the control without independent evidence (RJ 2026-11-08)
        assert system.count(DRAFT_ONLY_OLD) == 1
        system = system.replace(DRAFT_ONLY_OLD, DRAFT_ONLY_NEW)
    if cfg.gate.margin:
        assert system.count("at least zero") == 1 and system.count("GREEN >= 0") == 1
        system = system.replace("at least zero", f"at least {cfg.gate.margin}").replace("GREEN >= 0", f"GREEN >= {cfg.gate.margin}")
    if cfg.learner.anon:
        assert system.count(ANON_OLD) == 1
        system = system.replace(ANON_OLD, ANON_NEW)
    if cfg.learner.blind:                                            # the control without outcome feedback (RJ 2026-11-10)
        for old, new in BLIND:
            assert system.count(old) == 1, old
            system = system.replace(old, new)
    if cfg.learner.no_experience:
        assert system.count(NO_EXPERIENCE_OLD) == 1
        system = system.replace(NO_EXPERIENCE_OLD, NO_EXPERIENCE_NEW)
    return system


# ---- user prompt: grammar ----
EDIT_DOC = {"add": "finding silent in the draft: state it (a denial of it is corrected)",
            "remove": "finding stated: remove the statement, keeping anything else in the sentence",
            "rewrite_normal": "finding stated: rewrite to the normal statement (aortic dilation only)",
            "side": "pleural effusion stated with the wrong side: rewrite with the measured side",
            "add_value": "the structure is mentioned without a number: add its measured diameter",
            "type": "medical material stated without its type: add the device type from metal location",
            "add_typed": "medical material silent: add the typed device sentence instead of the generic one"}
NO_EVIDENCE = "none"
NO_EVIDENCE_EDITS = ("side", "type", "add_typed")                # these carry their own measurement, so they may fire without evidence


def grammar_text(evidence_names, edits_of, edits):
    """edits_of: abnormality -> its edits; edits: the edits of this evidence set"""
    no_evidence = f" The edits {', '.join(NO_EVIDENCE_EDITS)} may also use evidence '{NO_EVIDENCE}': the rule then fires on every eligible draft." if set(NO_EVIDENCE_EDITS) <= set(edits) else ""
    L = ["GRAMMAR. A rule = finding, edit, evidence, direction (>= or <=, either direction for any edit). The rule fires when the evidence is on "
         f"that side of a threshold, which code picks from the percentiles {', '.join(THRESHOLDS)} of that evidence over the drafts eligible for the "
         f"edit.{no_evidence}", "", "Edits:"]
    L += [f"- {e}: {EDIT_DOC[e]}" for e in edits]
    L += ["", "Evidence vocabulary (any evidence for any finding): " + ", ".join(evidence_names),
          "Findings and their edits: " + "; ".join(f"{f}: {edits}" for f, edits in edits_of.items())]
    return "\n".join(L)


# ---- user prompt: the initial bank ----
EVIDENCE_TEXT = {
    "draft_n_abnormalities": "number of the 18 CT-RATE abnormalities the draft states, from RadBERT on the draft's text",
    "draft_words": "number of words in the draft",
    "ctr": "cardiothoracic ratio (heart width / thorax width) from the heart and lung segmentation",
    "calc_coronary": "coronary artery calcium volume, mm3, from the CT inside the coronary-artery masks",
    "calc_aortic": "aortic wall calcium volume, mm3",
    "pericardial_ml": "fluid volume in the pericardial sac, ml, from segmentation",
    "pleural_ml": "pleural fluid volume (both sides), ml, from segmentation",
    "esophagus_area": "cross-sectional area of the distal esophagus near the hiatus, mm2",
    "metal_mm3": "volume of very dense voxels inside the body (metal, devices), mm3",
    "ascending_mm": "largest diameter of the ascending aorta, mm, from the aorta centreline",
    "liver_hu_neg": "minus the median liver attenuation, HU (higher = lower liver attenuation)",
    "coronal_range": "range of the spine's angle in the coronal plane along the vertebrae, degrees",
    "heart_ml": "heart volume, ml, from the heart mask",
    "heart_total_ml": "sum of the four chambers and the myocardium, ml, from the heart-chamber masks",
    "atrium_left_ml": "left atrium volume, ml",
    "atrium_right_ml": "right atrium volume, ml",
    "ventricle_left_ml": "left ventricle volume, ml",
    "ventricle_right_ml": "right ventricle volume, ml",
    "myocardium_ml": "myocardium volume, ml",
    "pulmonary_artery_ml": "volume of the pulmonary artery mask, ml",
    "coronary_agatston": "Agatston-like score of the coronary calcium",
    "arch_mm": "largest diameter of the aortic arch, mm, from the aorta centreline",
    "descending_mm": "largest diameter of the descending thoracic aorta, mm",
    "pa_mm": "largest diameter of the main pulmonary artery, mm",
    "calc_aortic_arch": "calcium volume in the aortic arch wall, mm3",
    "calc_aortic_desc": "calcium volume in the descending thoracic aortic wall, mm3",
    "aorta_lumen_hu": "mean attenuation inside the aortic lumen, HU (high when intravenous contrast was given)",
    "laa950": "share of lung-core voxels (lobes eroded by 2 voxels) below -950 HU (low-attenuation area)",
    "laa910": "share of lung voxels below -910 HU",
    "laa950_upper": "share of upper-lobe (both upper lobes) lung-core voxels below -950 HU",
    "lung_p15": "15th percentile of lung attenuation, HU",
    "lung_mean_hu": "mean lung attenuation, HU",
    "lung_haa": "share of lung voxels between -500 and -100 HU (high-attenuation area)",
    "lung_dense": "share of lung voxels above -100 HU (dense lung)",
    "haa_right_upper": "high-attenuation share of the right upper lobe (raw HU)",
    "haa_right_middle": "high-attenuation share of the right middle lobe (raw HU)",
    "haa_right_lower": "high-attenuation share of the right lower lobe (raw HU)",
    "haa_left_upper": "high-attenuation share of the left upper lobe (raw HU)",
    "haa_left_lower": "high-attenuation share of the left lower lobe (raw HU)",
    "dense_right_upper": "dense share of the right upper lobe (raw HU)",
    "dense_right_middle": "dense share of the right middle lobe (raw HU)",
    "dense_right_lower": "dense share of the right lower lobe (raw HU)",
    "dense_left_upper": "dense share of the left upper lobe (raw HU)",
    "dense_left_lower": "dense share of the left lower lobe (raw HU)",
    "lung_ml": "total lung volume, ml",
    "lung_lr_ratio": "left lung volume divided by right lung volume",
    "nodule_count": "number of lung-nodule components of 10-4200 mm3 (0 when any component exceeds 5000 mm3)",
    "nodule_max_mm3": "volume of the largest lung-nodule component, mm3",
    "esophagus_area_low30": "largest axial cross-section of the esophagus within its lowest 30 mm, mm2",
    "esophagus_vol_low60": "esophagus volume within its lowest 60 mm, ml",
    "esophagus_len_wide": "length of the lowest 80 mm of the esophagus with a cross-section above 250 mm2, mm",
    "esophagus_area_upper": "largest esophagus cross-section 60-120 mm above its lowest slice, mm2",
    "esophagus_gas_ml": "gas inside the lowest 60 mm of the esophagus, ml",
    "stomach_above_lungbase_ml": "stomach volume above the lung base, ml",
    "stomach_top_above_lungbase": "height of the stomach top above the lung base, mm",
    "liver_minus_spleen_hu": "liver attenuation minus spleen attenuation, HU",
    "spleen_hu": "median spleen attenuation, HU",
    "spleen_ml": "spleen volume in the field of view, ml",
    "liver_ml": "liver volume in the field of view, ml",
    "kidney_cyst_left_ml": "left kidney cyst volume, ml",
    "kidney_cyst_right_ml": "right kidney cyst volume, ml",
    "gallbladder_ml": "gallbladder volume in the field of view, ml",
    "thyroid_ml": "thyroid volume, ml",
    "spine_lateral_dev": "largest left-right distance of a vertebra centroid from the line through the first and last, mm",
    "spine_sagittal_range": "range of the spine's angle in the sagittal plane along the vertebrae, degrees",
    "vertebra_hu": "median attenuation of the vertebrae, HU",
    "metal_components": "number of internal metal components",
    "metal_largest_mm3": "volume of the largest internal metal component, mm3",
    "metal_external_mm3": "volume of metal outside the body proper (on the skin, clothing), mm3"}


def evidence_text(evidence_names, anon):
    L = ["EVIDENCE (the initial bank: every value is available for every scan):"]
    for z in evidence_names:
        if anon:
            L.append(f"- {z}: (no description)")
        else:
            L.append(f"- {z}: {evidence_line(z)}")
    return "\n".join(L)


# ---- user prompt: experience ----
def rule_text(rule):
    return f"{rule['finding']} | {rule['edit']} | " + (" and ".join(f"{z} {op} {threshold} ({q:g})" for z, op, threshold, q in rule["conds"]) or "no evidence")


def green_text(s):
    return s["dG"] if s["dG"] is None else round(s["dG"], 2)


def condition_text(probe):
    return f"{probe['evidence']} {probe['direction']}" if probe["evidence"] != NO_EVIDENCE else "no evidence"


def probe_line(h):
    """one probe in one line: the best threshold, and what the gate said"""
    p = h["raw"]
    lab = f"- [{h['id']}] {p['finding']} | {p['edit']} | {condition_text(p)}"
    if "reject" in h:
        return f"{lab}: rejected ({h['reject']})"
    pts = [x for x in h["sweep"] if "outcome" in x and x["outcome"] is not None]
    swept = " (11 thresholds swept)" if p["evidence"] != NO_EVIDENCE else ""
    if not pts:
        return f"{lab}{swept}: changes no fit draft"
    ok = [x for x in pts if x["refused"] is None]
    b = max(ok or pts, key=lambda x: x["outcome"]["dF1"])
    s = b["outcome"]
    at = f"best {b['threshold']} ({b['value']:g}): " if b["threshold"] else ""
    extra = (f", precision {100 * s['precision']:.0f}%" if "precision" in s else "") + (f", agreement {100 * s['agree']:.0f}%" if "agree" in s else "")
    return f"{lab}{swept}: {at}n {s['n']}, dF1 {s['dF1']:+.3f}, dGREEN {green_text(s)}{extra}, " + ("admitted" if ok else f"not admitted ({b['refused']})")


def blind_experience_text(probes):
    """the probes tried, without any outcome or bank (the control without outcome feedback)"""
    if not probes:
        return "NO PROBES YET"
    return "PROBES SO FAR (outcomes are not shown):\n" + "\n".join(f"- [{h['id']}] {h['raw']['finding']} | {h['raw']['edit']} | {condition_text(h['raw'])}" + (f": rejected ({h['reject']})" if "reject" in h else "") for h in probes)


def experience_text(evidence_names, probes, active, rnd, no_experience):
    """the bank as experience: use of each evidence, the active rules [(id, rule, outcome)], one line per probe"""
    if no_experience:
        last = [h for h in probes if h["round"] == rnd - 1]
        return "LAST ROUND (earlier rounds are not kept):\n" + "\n".join(probe_line(h) for h in last) if last else "NO OUTCOMES YET"
    use = collections.Counter(h["raw"]["evidence"] for h in probes)
    L = ["EXPERIENCE", "Probes per evidence so far: " + ", ".join(f"{z} {use[z]}" for z in evidence_names), "",
         f"ACTIVE RULES (bank after the gate, {len(active)}):"]
    L += [f"- [{i}] {rule_text(rule)}: fit n {s['n']}, dF1 {s['dF1']:+.3f}, dGREEN {green_text(s)}" for i, rule, s in active]
    if probes:
        L += ["", "PROBES SO FAR (each with its best threshold):"] + [probe_line(h) for h in probes]
    return "\n".join(L)


# ---- user prompt: the round ----
ROUND_TEXT = {"explore": "Propose probes not tried yet: findings and edits that have no admitted rule, evidence you have not used, or the other direction.",
              "explore_p2": "Submit probes you have not tried: finding and edit pairs without an accepted rule, unused evidence, or the opposite direction."}


def user_prompt(head, experience, rnd, rounds, budget, prompt):
    return head + "\n\n" + experience + "\n\n\n" + f"ROUND {rnd + 1} of {rounds}. Propose at most {budget} new probes. " + ROUND_TEXT[prompt]


def repeat_note(repeats, n):
    """added to the round's prompt when a reply repeated earlier probes"""
    lines = [f"- {p['finding']} | {p['edit']} | {condition_text(p)}" for p in repeats]
    return ("\n\nYOUR REPLY REPEATED THESE PROBES, ALREADY TRIED, SO THEY WERE NOT RUN:\n" + "\n".join(lines) +
            f"\nPropose {n} different probes: a finding and edit, an evidence or a direction you have not tried together.")


# ---- reject reasons, shown to the agent with the probe's outcome ----
REJECT = {"no_edit": "no edit '{e}' for '{f}' in the table (finding names and edits must match the grammar)",
          "no_evidence": "this edit needs an evidence",
          "evidence": "evidence must be one of {names}",
          "few_values": "{evidence} has fewer than 5 values among the eligible fit drafts"}


# ---- the review of a learned bank (rulebank/review.py): third wording, RJ 2026-11-08 (the second judged thresholds and refused indirect signs) ----
REVIEW_SYSTEM = ("You review rules that revise chest CT report drafts. A rule says: when one piece of independent image evidence is above or below a threshold, apply an edit for one finding. Two kinds of "
                 "evidence exist. A classifier value named after a finding is the probability, from image classifiers that read the scan, that the scan shows that finding: it is image evidence for that "
                 "finding and for no other. A measurement is a quantity measured on segmented anatomy. The rule was kept because it helped on development data, which is not enough: an association can "
                 "be an accident. Judge only the medical link: does this evidence bear on this finding, and does the direction make sense (more evidence for adding a finding or a value, less for removing "
                 "one)? The evidence need not measure the finding itself: a recognised indirect sign of the finding is a valid link. A rule fails only when the evidence concerns a different finding or "
                 "structure with no recognised relation to this one, or when the direction is reversed. The threshold is chosen elsewhere and is not shown: do not judge it, and do not judge whether the "
                 "edit is worthwhile or how accurate the evidence is. Say why the rule should hold or fail, then decide. Answer with JSON only.")


REVIEW_DRAFT_SYSTEM = ("You review rules that revise chest CT report drafts. A rule says: when one signal computed from the text of the draft is above or below a threshold, apply an edit for one finding. Two kinds of "
                       "signal exist. A value named after a finding is the probability, from a text classifier reading the draft, that the draft states that finding: it says what the draft reports and nothing about the "
                       "image. The other values count what the draft states or how long it is. The rule was kept because it helped on development data, which is not enough: an association can "
                       "be an accident. Judge only the medical link: does this signal bear on this finding, and does the direction make sense (a stronger related statement for adding a finding or a value, a weaker one for removing "
                       "one)? The signal need not be the finding itself: a recognised clinical association between two findings is a valid link. A rule fails only when the signal concerns a different finding or "
                       "structure with no recognised relation to this one, when it has no medical relation to the finding at all, or when the direction is reversed. The threshold is chosen elsewhere and is not shown: do not judge it, and do not judge whether the "
                       "edit is worthwhile or how accurate the signal is. Say why the rule should hold or fail, then decide. Answer with JSON only.")


def review_system(cfg):
    return REVIEW_DRAFT_SYSTEM if cfg.evidence == "draft_only" else REVIEW_SYSTEM


def review_text(rule):
    """a rule as the reviewer reads it: no scores and no threshold, so one (finding, edit, evidence, direction) gets one verdict in every fold"""
    edit = f"{rule['edit']}: {EDIT_DOC[rule['edit']]}"
    if not rule["conds"]:
        return f"Finding: {rule['finding']}\nEdit: {edit}\nCondition: none, the edit applies to every eligible draft."
    z, op, _, _ = rule["conds"][0]
    return f"Finding: {rule['finding']}\nEdit: {edit}\nCondition: {z} {op} a threshold, where {z} is the {evidence_line(z)}."


# ---- the editor agent (rulebank/editor.py): from main_v5 reportlevel/prompts_v14, one change: additions always go to the findings ----
EDITOR_SYSTEM = (
    "You revise a chest CT report that an AI report generator drafted. You receive:\n"
    "1. the draft, as numbered sentences, each marked (findings) or (impression), with the report's sub-section name when it has one;\n"
    "2. an evidence sheet computed from the same CT by tools that are independent of the report generator: image classifiers\n"
    "   that give a probability per abnormality, and automatic measurements;\n"
    "3. rules, each with an id.\n\n"
    "Apply the rules exactly as written:\n"
    "- change the report only where a rule applies, and there do exactly what the rule says; when no rule applies, leave the\n"
    "  report unchanged;\n"
    "- never change, reorder, merge or drop any other sentence;\n"
    "- when a rule gives the sentence to write, write it word for word;\n"
    "- write no device type, side, lobe, segment, size, number or severity that neither the draft nor the evidence sheet gives;\n"
    "- never let the report contradict itself.\n"
    "Answer with the list of your edits (empty when no rule applies). Each edit names the one rule it applies (rule_id) and is\n"
    "one of: rewrite (sentence_id = the number of the draft sentence, new_text = the full new sentence), remove (sentence_id,\n"
    "new_text empty), add (sentence_id 0, new_text = the new sentence, placed in the findings). Write new_text as\n"
    "report text only: no sentence number, no section tag, no evidence values other than the ones a rule asks for.")
EDITOR_DRAFT_OLD = ("2. an evidence sheet computed from the same CT by tools that are independent of the report generator: image classifiers\n"
                    "   that give a probability per abnormality, and automatic measurements;\n")
EDITOR_DRAFT_NEW = "2. a sheet of signals computed from the text of the draft itself: a text classifier's probability per abnormality, and counts of the draft's content;\n"
assert EDITOR_SYSTEM.count(EDITOR_DRAFT_OLD) == 1
EDITOR_DRAFT_SYSTEM = EDITOR_SYSTEM.replace(EDITOR_DRAFT_OLD, EDITOR_DRAFT_NEW)         # the control without independent evidence (RJ 2026-11-08)
assert EDITOR_SYSTEM.count("3. rules, each with an id.\n") == 1 and EDITOR_SYSTEM.count("Apply the rules exactly as written:\n") == 1
TEXT_MEMORY_SYSTEM = (EDITOR_SYSTEM.replace("3. rules, each with an id.\n", "3. rules, each with an id, an evidence condition and a draft condition.\n")      # the text memory control (baselines/text_memory.py), RJ 2026-11-08
                      .replace("Apply the rules exactly as written:\n", "Decide for each rule whether it applies: compare the value on the evidence sheet with the rule's threshold and check the draft condition. "
                               "A rule applies only when both hold, and a value marked not available never satisfies a condition. When two rules add the same abnormality, apply only the first.\n"
                               "Apply the rules that apply exactly as written:\n"))
# the direct LLM control (baselines/direct_llm.py), RJ 2026-11-08: main_v5 reportlevel/prompts_free.py SYSTEM and stage_b/sentences.py HAND; one change: additions always go to the findings, as for the editor
CTRATE_SENTENCES = {"Medical material": ("A medical device is observed.", "No medical device is observed."),
        "Arterial wall calcification": ("Atheroma plaques are observed in the aorta.", "There are no atheromatous plaques in the aorta."),
        "Cardiomegaly": ("The heart is larger than normal.", "Heart contour and size are normal."),
        "Pericardial effusion": ("Pericardial effusion is observed.", "No pericardial effusion was detected."),
        "Coronary artery wall calcification": ("Calcific atheroma plaques are observed in the coronary arteries.", "No calcific plaques are observed in the coronary arteries."),
        "Hiatal hernia": ("There is a hiatal hernia.", "No hiatal hernia was detected."),
        "Lymphadenopathy": ("There are pathologically enlarged lymph nodes in the mediastinum.", "No pathologically enlarged lymph nodes were detected in the mediastinum and hilar regions."),
        "Emphysema": ("There are emphysematous changes in both lungs.", "No emphysematous changes were detected in both lungs."),
        "Atelectasis": ("There are linear atelectasis in both lungs.", "No atelectasis was detected in both lungs."),
        "Lung nodule": ("There are millimetric nonspecific nodules in both lungs.", "No nodule was detected in both lungs."),
        "Lung opacity": ("There are ground glass densities in both lungs.", "No ground glass density was detected in both lungs."),
        "Pulmonary fibrotic sequela": ("There are pleuroparenchymal sequelae changes in both lungs.", "No sequelae changes were detected in both lungs."),
        "Pleural effusion": ("Minimal pleural effusion is observed.", "No pleural effusion was detected."),
        "Mosaic attenuation pattern": ("A mosaic attenuation pattern was observed in both lungs.", "No mosaic attenuation pattern was detected."),
        "Peribronchial thickening": ("Peribronchial thickenings are observed in both lungs.", "No peribronchial thickening was detected."),
        "Consolidation": ("There are consolidation areas in both lungs.", "No consolidation was detected in both lungs."),
        "Bronchiectasis": ("There are bronchiectasis in both lungs.", "No bronchiectasis was detected in both lungs."),
        "Interlobular septal thickening": ("Interlobular septal thickenings are observed in both lungs.", "No interlobular septal thickening was detected.")}
DIRECT_LLM_SYSTEM = """You revise a chest CT report that an AI report generator drafted. You receive:
1. the draft, as numbered sentences, each marked (findings) or (impression), with the report's sub-section name when it has one;
2. an evidence sheet computed from the same CT by tools that are independent of the report generator: image classifiers
   that give a probability per abnormality, and automatic measurements.

Correct the draft where the evidence clearly warrants it: add an abnormality the draft omits when the evidence for it is
strong, remove or rewrite a sentence the evidence clearly contradicts, and otherwise leave the draft unchanged. Use your
own judgement about what counts as strong evidence; the classifiers and measurements can be wrong.
- never change, reorder, merge or drop any other sentence;
- write new sentences in the plain style of the draft (e.g. "There are emphysematous changes in both lungs.", "Pericardial
  effusion is observed.", "The heart is larger than normal.");
- write no device type, side, lobe, segment, size, number or severity that neither the draft nor the evidence sheet gives;
- never let the report contradict itself.
When you add or rewrite a sentence, use the CT-RATE phrasing of the abnormality (present / absent):
{sentences}
Answer with the list of your edits (empty when nothing warrants a change). Each edit is one of: rewrite (sentence_id = the
number of the draft sentence, new_text = the full new sentence), remove (sentence_id, new_text empty), add (sentence_id 0,
new_text = the new sentence, placed in the findings). rule_id is always "direct". Write new_text as report text
only: no sentence number, no section tag.""".format(sentences="\n".join(f'- {l}: "{a}" / "{b}"' for l, (a, b) in CTRATE_SENTENCES.items()))
ADDED = {**ADD_SENT, "hepatic steatosis": LIVER_SENT, "aortic dilation": "The ascending aorta is dilated, measuring <ascending mm> mm."}
ASCENDING_MM, SIDE, DEVICE_SENTENCE, VALUE_SENTENCE = "<ascending mm>", "<side>", "<device sentence>", "<value sentence>"     # filled in per draft
EDITOR_RULE = {
    "add": "Add the sentence '{sentence}' to the report: this rule applies even when no sentence mentions {finding}. If sentences deny {finding}, none may do so any more: a sentence that denies only "
           "that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only {finding}. "
           "The sentence above appears exactly once in the report.",
    "remove": "The report must no longer state {finding}. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, "
              "keep them and drop only this one; delete a sentence only when nothing else in it remains.",
    "remove_heart": " If no statement about the heart is left, the findings must state 'Heart contour and size are normal.'",
    "rewrite_normal": "The thoracic aorta is normal in calibre. Remove every statement of aortic dilatation, ectasia, aneurysm or a wider-than-normal aorta, also inside list sentences of "
                      "the impression, and make the findings state 'Thoracic aorta diameter is normal.'",
    "side": f"The pleural effusion is {SIDE}. Rewrite the report's pleural effusion statement with this side and keep any size word; keep the other findings of the sentence.",
    "append": "Add '{sentence}' to the findings."}


def do_text(finding, edit):
    """the instruction of a rule; the parts in angle brackets are filled in per draft"""
    if edit == "add":
        return EDITOR_RULE["add"].format(finding=finding, sentence=ADDED[finding])
    if edit == "add_typed":
        return EDITOR_RULE["add"].format(finding=finding, sentence=DEVICE_SENTENCE)
    if edit == "remove":
        return EDITOR_RULE["remove"].format(finding=finding) + (EDITOR_RULE["remove_heart"] if finding == "Cardiomegaly" else "")
    if edit in ("rewrite_normal", "side"):
        return EDITOR_RULE[edit]
    return EDITOR_RULE["append"].format(sentence=VALUE_SENTENCE if edit == "add_value" else DEVICE_SENTENCE)


def evidence_line(z):
    if z.startswith("draft:"):
        return f"probability, from RadBERT reading the draft's own text, that the draft states {z[6:]}"
    return f"mean probability of independent image classifiers that the scan shows {z[4:]}" if z.startswith("clf:") else EVIDENCE_TEXT[z]


def editor_prompt(numbered_text, sheet, rules):
    return f"Draft report:\n{numbered_text}\n\nEvidence sheet:\n{sheet}\n\nRules:\n{rules}"


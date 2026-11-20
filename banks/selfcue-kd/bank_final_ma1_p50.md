# Rule bank final_ma1_p50 (30 rules; generated, do not edit)

## R1 · Cardiomegaly · remove

- When (code): clf:Cardiomegaly <= 0.7805
- Do: The report must no longer state Cardiomegaly. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains. If no statement about the heart is left, the findings must state 'Heart contour and size are normal.'
- Why: The classifier value clf:Cardiomegaly directly estimates the probability that the scan shows cardiomegaly. A low probability (≤ threshold) indicates insufficient image evidence for cardiomegaly, so removing the statement is medically appropriate.
- Record: {'n': 76, 'dF1': 0.1822536772290917, 'dG': 4.525755247504475, 'precision': 0.15789473684210525}

## R2 · Pleural effusion · add

- When (code): clf:Pleural effusion >= 0.8727
- Do: Add the sentence 'Minimal pleural effusion is observed.' to the report: this rule applies even when no sentence mentions Pleural effusion. If sentences deny Pleural effusion, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Pleural effusion. The sentence above appears exactly once in the report.
- Why: The classifier probability is direct image evidence for pleural effusion, and a high value correctly indicates more evidence for adding the finding.
- Record: {'n': 67, 'dF1': 0.3546344908659762, 'dG': 0.3156006514215468, 'precision': 0.7910447761194029}

## R3 · Pericardial effusion · add

- When (code): pericardial_ml >= 34.1272
- Do: Add the sentence 'Minimal pericardial effusion is observed.' to the report: this rule applies even when no sentence mentions Pericardial effusion. If sentences deny Pericardial effusion, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Pericardial effusion. The sentence above appears exactly once in the report.
- Why: Pericardial fluid volume measured from segmentation directly quantifies the amount of fluid in the pericardial sac, which is the defining characteristic of a pericardial effusion. A volume above the threshold provides direct evidence for the presence of an effusion, so the direction (higher volume → add the finding) is appropriate.
- Record: {'n': 66, 'dF1': 0.47989279338202095, 'dG': -0.595153385928787, 'precision': 0.6818181818181818}

## R4 · Pericardial effusion · remove

- When (code): pericardial_ml <= 15.8806
- Do: The report must no longer state Pericardial effusion. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: The measurement pericardial_ml directly quantifies fluid in the pericardial sac, which is the defining characteristic of a pericardial effusion. A low volume (≤ the threshold) indicates little or no fluid, supporting the removal of a stated pericardial effusion finding. The evidence is directly related to the finding and the direction (low volume → remove) is appropriate.
- Record: {'n': 35, 'dF1': 0.06041560842459148, 'dG': 1.376060447489019, 'precision': 0.11428571428571428}

## R5 · ascending aorta · add_value

- When (code): clf:Arterial wall calcification >= 0.8152
- Do: Add '<value sentence>' to the findings.
- Why: Arterial wall calcification is a recognized sign of aortic disease and directly involves the aortic wall, including the ascending aorta. A high classifier probability therefore supports the need to report a quantitative measurement (diameter) of the ascending aorta, not the opposite. The evidence is relevant and the direction (adding a value) is appropriate.
- Record: {'n': 475, 'dF1': -0.02271214418070855, 'dG': 0.22064752281328837}

## R6 · descending aorta · add_value

- When (code): descending_mm >= 28.4605
- Do: Add '<value sentence>' to the findings.
- Why: The evidence is a direct measurement of the descending thoracic aorta diameter, which is the same structure referenced by the finding. This measurement is a recognized image evidence for the descending aorta and therefore bears on the finding. The rule’s direction—adding the numeric diameter when the measured value meets or exceeds a threshold—makes sense because a larger (or at‑threshold) diameter provides a clear reason to report the value, especially when the report currently mentions the aorta without a number. No unrelated structure or reversed logic is involved.
- Record: {'n': 478, 'dF1': -0.017129203108517288, 'dG': 0.07007355163554647}

## R7 · main pulmonary artery · add_value

- When (code): every eligible draft
- Do: Add '<value sentence>' to the findings.
- Why: not reviewed: no condition, or an edit that carries its own image evidence
- Record: {'n': 33, 'dF1': 0.0, 'dG': 1.4468498425717673}

## R8 · Arterial wall calcification · add

- When (code): clf:Arterial wall calcification >= 0.7833
- Do: Add the sentence 'Atheroma plaques are observed in the aorta.' to the report: this rule applies even when no sentence mentions Arterial wall calcification. If sentences deny Arterial wall calcification, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Arterial wall calcification. The sentence above appears exactly once in the report.
- Why: The classifier clf:Arterial wall calcification directly estimates the probability that arterial wall calcification is present in the scan. A high classifier value (>= threshold) is evidence supporting the presence of this finding, so adding a statement that the finding is present (instead of denying it) is medically appropriate. The evidence is directly linked to the same finding and the direction (more evidence → add) is correct.
- Record: {'n': 113, 'dF1': 0.1649425456300122, 'dG': 0.9625735621310847, 'precision': 0.6371681415929203}

## R9 · Lung nodule · add

- When (code): clf:Lung nodule >= 0.3763
- Do: Add the sentence 'There are millimetric nonspecific nodules in both lungs.' to the report: this rule applies even when no sentence mentions Lung nodule. If sentences deny Lung nodule, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Lung nodule. The sentence above appears exactly once in the report.
- Why: The classifier value clf:Lung nodule directly estimates the probability that a lung nodule is present in the scan. This is independent image evidence for the same finding, and a higher probability logically supports adding the finding. The direction of the rule (add when probability is above the threshold) matches the medical implication of the evidence.
- Record: {'n': 1867, 'dF1': 2.2547839899946496, 'dG': 0.26937814951156114, 'precision': 0.5281199785752544}

## R10 · Medical material · add_typed

- When (code): pericardial_ml >= 0.0525
- Do: Add the sentence '<device sentence>' to the report: this rule applies even when no sentence mentions Medical material. If sentences deny Medical material, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Medical material. The sentence above appears exactly once in the report.
- Why: not reviewed: no condition, or an edit that carries its own image evidence
- Record: {'n': 209, 'dF1': 0.7396449651887522, 'dG': 0.13072617672881326, 'precision': 0.6172248803827751}

## R11 · Emphysema · add

- When (code): clf:Emphysema >= 0.7389
- Do: Add the sentence 'There are emphysematous changes in both lungs.' to the report: this rule applies even when no sentence mentions Emphysema. If sentences deny Emphysema, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Emphysema. The sentence above appears exactly once in the report.
- Why: The classifier value clf:Emphysema directly measures the probability that the scan shows emphysema, which is the exact finding in question. A higher probability logically supports adding the finding to the report, so the evidence is medically relevant and the direction (more evidence → add) is correct.
- Record: {'n': 129, 'dF1': 0.42092211576146354, 'dG': 0.6096726650762221, 'precision': 0.5503875968992248}

## R12 · Emphysema · remove

- When (code): clf:Emphysema <= 0.6762
- Do: The report must no longer state Emphysema. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: The classifier probability directly measures the presence of emphysema. A low probability (<= threshold) indicates less evidence for emphysema, which justifies removing the finding statement. The evidence is directly linked to the finding and the direction (less evidence → removal) is appropriate.
- Record: {'n': 124, 'dF1': 0.10427575120647958, 'dG': 3.127164120404158, 'precision': 0.16129032258064516}

## R13 · Cardiomegaly · add

- When (code): clf:Cardiomegaly >= 0.7749
- Do: Add the sentence 'The heart size is increased, consistent with cardiomegaly.' to the report: this rule applies even when no sentence mentions Cardiomegaly. If sentences deny Cardiomegaly, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Cardiomegaly. The sentence above appears exactly once in the report.
- Why: The classifier probability directly estimates the presence of cardiomegaly, so a high value provides direct evidence supporting the addition of this finding.
- Record: {'n': 131, 'dF1': 0.37108610615874227, 'dG': -0.10164131527542318, 'precision': 0.5267175572519084}

## R14 · Atelectasis · add

- When (code): clf:Atelectasis >= 0.5939
- Do: Add the sentence 'There are linear atelectasis in both lungs.' to the report: this rule applies even when no sentence mentions Atelectasis. If sentences deny Atelectasis, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Atelectasis. The sentence above appears exactly once in the report.
- Why: The classifier probability directly measures the presence of atelectasis, which is the exact finding in question. A high probability (above the threshold) provides evidence supporting the presence of atelectasis, so the direction of the edit (adding the finding) is appropriate.
- Record: {'n': 516, 'dF1': 1.0597445491705244, 'dG': -0.2808912335561122, 'precision': 0.5329457364341085}

## R15 · Bronchiectasis · remove

- When (code): clf:Bronchiectasis <= 0.6856
- Do: The report must no longer state Bronchiectasis. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: The classifier probability directly estimates the presence of bronchiectasis. A low probability (≤ threshold) indicates less evidence for the finding, which correctly supports removing the bronchiectasis statement.
- Record: {'n': 28, 'dF1': 0.014564237999067586, 'dG': 4.516069248212104, 'precision': 0.14285714285714285}

## R16 · Consolidation · add

- When (code): clf:Consolidation >= 0.7802
- Do: Add the sentence 'There are consolidation areas in both lungs.' to the report: this rule applies even when no sentence mentions Consolidation. If sentences deny Consolidation, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Consolidation. The sentence above appears exactly once in the report.
- Why: The classifier clf:Consolidation directly estimates the probability that the scan shows consolidation, which is the exact finding in question. A high classifier value (above the threshold) provides image evidence supporting the presence of consolidation, so the direction of the rule (adding the finding) aligns with the evidence.
- Record: {'n': 121, 'dF1': 0.29474429501215127, 'dG': -0.9102156823352736, 'precision': 0.5950413223140496}

## R17 · Coronary artery wall calcification · add

- When (code): calc_coronary >= 26.58
- Do: Add the sentence 'Calcific atheroma plaques are observed in the coronary arteries.' to the report: this rule applies even when no sentence mentions Coronary artery wall calcification. If sentences deny Coronary artery wall calcification, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Coronary artery wall calcification. The sentence above appears exactly once in the report.
- Why: Coronary artery calcium volume directly quantifies calcification in the coronary artery wall, which is the finding in question. A value above the threshold indicates the presence of calcification, so the evidence supports adding the finding.
- Record: {'n': 230, 'dF1': 0.2928859636971737, 'dG': 0.24168815696948692, 'precision': 0.5695652173913044}

## R18 · Lung opacity · add

- When (code): clf:Lung opacity >= 0.4812
- Do: Add the sentence 'There are ground glass densities in both lungs.' to the report: this rule applies even when no sentence mentions Lung opacity. If sentences deny Lung opacity, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Lung opacity. The sentence above appears exactly once in the report.
- Why: The classifier probability clf:Lung opacity directly measures the likelihood that the scan shows a lung opacity, which is the exact finding in question. A high probability therefore provides evidence supporting the presence of the finding, making the direction of the edit (adding the finding) appropriate.
- Record: {'n': 461, 'dF1': 0.5423315949936267, 'dG': -0.5552056569329558, 'precision': 0.5683297180043384}

## R19 · Lymphadenopathy · add

- When (code): clf:Lymphadenopathy >= 0.771
- Do: Add the sentence 'Enlarged lymph nodes are observed in the mediastinum.' to the report: this rule applies even when no sentence mentions Lymphadenopathy. If sentences deny Lymphadenopathy, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Lymphadenopathy. The sentence above appears exactly once in the report.
- Why: The classifier value clf:Lymphadenopathy directly measures the probability that the scan shows lymphadenopathy, which is the exact finding in question. A higher probability (above the threshold) provides evidence supporting the presence of lymphadenopathy, so adding the finding is appropriate. The evidence is not about a different structure and the direction (more evidence → add) is correct.
- Record: {'n': 133, 'dF1': 0.5060169126539762, 'dG': -0.008424577432235833, 'precision': 0.6165413533834586}

## R20 · Mosaic attenuation pattern · remove

- When (code): clf:Mosaic attenuation pattern <= 0.8659
- Do: The report must no longer state Mosaic attenuation pattern. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: The classifier probability directly measures the presence of a mosaic attenuation pattern. A low probability (<= threshold) indicates less evidence for the finding, so removing the statement is medically appropriate. The evidence is directly linked to the finding and the direction (lower value → removal) makes sense.
- Record: {'n': 26, 'dF1': 0.03313646263055148, 'dG': 6.333111333111333, 'precision': 0.19230769230769232}

## R21 · Pulmonary fibrotic sequela · add

- When (code): clf:Pulmonary fibrotic sequela >= 0.6955
- Do: Add the sentence 'There are pleuroparenchymal sequelae changes in both lungs.' to the report: this rule applies even when no sentence mentions Pulmonary fibrotic sequela. If sentences deny Pulmonary fibrotic sequela, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Pulmonary fibrotic sequela. The sentence above appears exactly once in the report.
- Why: The classifier clf:Pulmonary fibrotic sequela directly quantifies the likelihood that the scan exhibits Pulmonary fibrotic sequela. A value above the threshold provides evidence for the presence of this finding, so adding it to the report is appropriate. The evidence pertains to the same finding and the direction (higher probability → add) is correct.
- Record: {'n': 278, 'dF1': 1.0149889967937256, 'dG': 0.3007111202814701, 'precision': 0.5755395683453237}

## R22 · Medical material · add

- When (code): clf:Medical material >= 0.7839
- Do: Add the sentence 'A medical device is observed.' to the report: this rule applies even when no sentence mentions Medical material. If sentences deny Medical material, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Medical material. The sentence above appears exactly once in the report.
- Why: The classifier named 'Medical material' directly estimates the probability that the scan shows that finding, providing independent image evidence for it. A high classifier value therefore supports adding the finding, matching the intended direction of the rule.
- Record: {'n': 292, 'dF1': 1.2842491268546112, 'dG': -0.32329816967365205, 'precision': 0.6027397260273972}

## R23 · Medical material · type

- When (code): every eligible draft
- Do: Add '<device sentence>' to the findings.
- Why: not reviewed: no condition, or an edit that carries its own image evidence
- Record: {'n': 61, 'dF1': 0.006459786204324303, 'dG': 1.1992091536450267, 'precision': 0.8360655737704918}

## R24 · Pleural effusion · side

- When (code): every eligible draft
- Do: The pleural effusion is <side>. Rewrite the report's pleural effusion statement with this side and keep any size word; keep the other findings of the sentence.
- Why: not reviewed: no condition, or an edit that carries its own image evidence
- Record: {'n': 44, 'dF1': -0.009619969383400928, 'dG': 1.2578033968408304, 'precision': 0.8409090909090909}

## R25 · Hiatal hernia · add

- When (code): esophagus_area >= 463.113
- Do: Add the sentence 'Sliding type hiatal hernia was observed at the lower end of the esophagus.' to the report: this rule applies even when no sentence mentions Hiatal hernia. If sentences deny Hiatal hernia, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Hiatal hernia. The sentence above appears exactly once in the report.
- Why: The cross‑sectional area of the distal esophagus near the hiatus is an indirect imaging sign of a hiatal hernia—when the hiatus is enlarged, the esophageal lumen often expands. Therefore a measurement that meets or exceeds the threshold provides evidence supporting the presence of a hiatal hernia, and the direction (adding the finding) aligns with that evidence.
- Record: {'n': 294, 'dF1': 1.9913767543379208, 'dG': -0.007626512891662008, 'precision': 0.5034013605442177}

## R26 · aortic dilation · rewrite_normal

- When (code): every eligible draft
- Do: The thoracic aorta is normal in calibre. Remove every statement of aortic dilatation, ectasia, aneurysm or a wider-than-normal aorta, also inside list sentences of the impression, and make the findings state 'Thoracic aorta diameter is normal.'
- Why: not reviewed: no condition, or an edit that carries its own image evidence
- Record: {'n': 123, 'dF1': -0.007364134088568619, 'dG': 1.7015250129216817, 'agree': 0.23577235772357724}

## R27 · hepatic steatosis · add

- When (code): liver_hu_neg >= -40.95
- Do: Add the sentence 'Hepatosteatosis.' to the report: this rule applies even when no sentence mentions hepatic steatosis. If sentences deny hepatic steatosis, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only hepatic steatosis. The sentence above appears exactly once in the report.
- Why: Liver attenuation measured in HU is a recognized imaging marker for hepatic steatosis; lower attenuation (i.e., a higher value of the negative median attenuation, liver_hu_neg) indicates more fat in the liver. The condition liver_hu_neg >= threshold therefore provides evidence that steatosis is present, supporting the addition of the finding when it was previously omitted. The evidence is directly linked to the finding and the direction (higher liver_hu_neg → more likely steatosis) aligns with the intended edit.
- Record: {'n': 144, 'dF1': 0.0, 'dG': 1.2490487984715282, 'agree': 0.6319444444444444}

## R28 · Pleural effusion · remove

- When (code): clf:Pleural effusion <= 0.9701
- Do: The report must no longer state Pleural effusion. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: The classifier probability is direct image evidence for pleural effusion. A value below the threshold indicates insufficient evidence for the finding, so removing the statement is medically appropriate.
- Record: {'n': 21, 'dF1': 0.04391164728281893, 'dG': 1.4386407243550101, 'precision': 0.2857142857142857}

## R29 · Coronary artery wall calcification · remove

- When (code): calc_coronary <= 6.72
- Do: The report must no longer state Coronary artery wall calcification. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: Calc_coronary is a direct measurement of calcium volume in the coronary artery wall, which is the anatomical substrate of coronary artery wall calcification. A value at or below the threshold indicates minimal or no calcification, so removing the statement that calcification is present is medically appropriate.
- Record: {'n': 69, 'dF1': 0.04243410356948374, 'dG': 0.9789810853211041, 'precision': 0.2898550724637681}

## R30 · hepatic steatosis · remove

- When (code): liver_hu_neg <= -35
- Do: The report must no longer state hepatic steatosis. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: Liver attenuation is a recognized indirect sign of hepatic steatosis. Lower attenuation (lower HU) indicates more fat, while higher attenuation suggests the liver is not fatty. liver_hu_neg is defined as the negative of the median liver attenuation, so a lower (more negative) liver_hu_neg corresponds to higher attenuation (less fat). The condition liver_hu_neg <= threshold therefore captures cases where attenuation is relatively high, i.e., evidence against steatosis. Removing the hepatic steatosis statement when this condition is met is medically appropriate.
- Record: {'n': 61, 'dF1': 0.0, 'dG': 4.139006284618628, 'agree': 0.3442622950819672}

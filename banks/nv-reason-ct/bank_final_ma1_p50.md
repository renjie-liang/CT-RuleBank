# Rule bank final_ma1_p50 (13 rules; generated, do not edit)

## R1 · Cardiomegaly · add

- When (code): clf:Cardiomegaly >= 0.9437
- Do: Add the sentence 'The heart size is increased, consistent with cardiomegaly.' to the report: this rule applies even when no sentence mentions Cardiomegaly. If sentences deny Cardiomegaly, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Cardiomegaly. The sentence above appears exactly once in the report.
- Why: The classifier probability is direct image evidence for Cardiomegaly, and a higher value (above the threshold) correctly supports adding the finding. The evidence pertains to the same finding and the direction (more evidence → add) is appropriate.
- Record: {'n': 70, 'dF1': 0.5027864910288145, 'dG': 0.842548324691182, 'precision': 0.7142857142857143}

## R2 · Coronary artery wall calcification · add

- When (code): calc_coronary >= 1277.1
- Do: Add the sentence 'Calcific atheroma plaques are observed in the coronary arteries.' to the report: this rule applies even when no sentence mentions Coronary artery wall calcification. If sentences deny Coronary artery wall calcification, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Coronary artery wall calcification. The sentence above appears exactly once in the report.
- Why: Coronary artery calcium volume directly quantifies calcification in the coronary artery wall, which is the finding in question. A value above the threshold indicates the presence of calcification, so the direction of the evidence (more evidence → add the finding) is appropriate.
- Record: {'n': 24, 'dF1': 0.014632329200296113, 'dG': 5.264078419449264, 'precision': 0.5}

## R3 · ascending aorta · add_value

- When (code): ascending_mm >= 40.45
- Do: Add '<value sentence>' to the findings.
- Why: The evidence is a direct measurement of the ascending aorta (ascending_mm), which is the same structure as the finding. A large diameter (>= threshold) is relevant to reporting the aortic size, so the direction (adding the measured value when the measurement is high) is appropriate. No unrelated structure or reversed direction is involved.
- Record: {'n': 95, 'dF1': 0.00040030938397705995, 'dG': 0.23337474950378173}

## R4 · hepatic steatosis · add

- When (code): liver_hu_neg >= -36
- Do: Add the sentence 'Hepatosteatosis.' to the report: this rule applies even when no sentence mentions hepatic steatosis. If sentences deny hepatic steatosis, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only hepatic steatosis. The sentence above appears exactly once in the report.
- Why: Liver attenuation measured in HU is a recognized quantitative sign of hepatic steatosis; lower attenuation (higher liver_hu_neg) indicates more fat. The rule uses a threshold on liver_hu_neg to trigger adding the finding, which aligns with the medical relationship and direction (higher liver_hu_neg supports presence of steatosis).
- Record: {'n': 153, 'dF1': 0.0022002359180959274, 'dG': 0.4461059680206168, 'agree': 0.8169934640522876}

## R5 · Lung opacity · add

- When (code): clf:Lung opacity >= 0.6261
- Do: Add the sentence 'There are ground glass densities in both lungs.' to the report: this rule applies even when no sentence mentions Lung opacity. If sentences deny Lung opacity, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Lung opacity. The sentence above appears exactly once in the report.
- Why: The classifier value clf:Lung opacity directly estimates the probability that the scan shows a lung opacity, which is the exact finding in question. A high probability (above the threshold) is medically consistent with the presence of a lung opacity, so the direction of the rule (adding the finding) is appropriate.
- Record: {'n': 162, 'dF1': 0.11133978809274936, 'dG': 0.4895148527224, 'precision': 0.5123456790123457}

## R6 · Pleural effusion · side

- When (code): every eligible draft
- Do: The pleural effusion is <side>. Rewrite the report's pleural effusion statement with this side and keep any size word; keep the other findings of the sentence.
- Why: not reviewed: no condition, or an edit that carries its own image evidence
- Record: {'n': 41, 'dF1': -0.01830798842847514, 'dG': 0.0264665889665892, 'precision': 0.8780487804878049}

## R7 · Medical material · add_typed

- When (code): every eligible draft
- Do: Add the sentence '<device sentence>' to the report: this rule applies even when no sentence mentions Medical material. If sentences deny Medical material, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Medical material. The sentence above appears exactly once in the report.
- Why: not reviewed: no condition, or an edit that carries its own image evidence
- Record: {'n': 80, 'dF1': 0.04073764233540089, 'dG': 1.2296921488097956, 'precision': 0.375}

## R8 · Arterial wall calcification · add

- When (code): clf:Arterial wall calcification >= 0.7918
- Do: Add the sentence 'Atheroma plaques are observed in the aorta.' to the report: this rule applies even when no sentence mentions Arterial wall calcification. If sentences deny Arterial wall calcification, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Arterial wall calcification. The sentence above appears exactly once in the report.
- Why: The classifier probability directly measures the presence of arterial wall calcification, which is the exact finding in question. A high probability (above the threshold) provides evidence supporting the presence of this finding, so adding it to the report is appropriate.
- Record: {'n': 110, 'dF1': 0.07163109654727862, 'dG': 0.025161536599445204, 'precision': 0.5272727272727272}

## R9 · Atelectasis · remove

- When (code): clf:Atelectasis <= 0.4812
- Do: The report must no longer state Atelectasis. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: The classifier value clf:Atelectasis directly measures the probability that the scan shows atelectasis. A low probability (below the threshold) provides evidence against the presence of atelectasis, so removing the statement is medically appropriate. The evidence is directly linked to the finding and the direction (lower value → less evidence) is correct.
- Record: {'n': 70, 'dF1': 0.03858335684523695, 'dG': 5.79079650508222, 'precision': 0.24285714285714285}

## R10 · Emphysema · remove

- When (code): clf:Emphysema <= 0.3485
- Do: The report must no longer state Emphysema. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: The classifier value clf:Emphysema directly measures the probability that the scan shows emphysema, which is the exact finding in question. A low probability (<= threshold) indicates less evidence for emphysema, so removing the statement is appropriate. The evidence is directly linked to the finding and the direction (lower probability → removal) makes sense.
- Record: {'n': 34, 'dF1': 0.08909960168479358, 'dG': 9.317014684661745, 'precision': 0.14705882352941177}

## R11 · Pulmonary fibrotic sequela · add

- When (code): clf:Pulmonary fibrotic sequela >= 0.7292
- Do: Add the sentence 'There are pleuroparenchymal sequelae changes in both lungs.' to the report: this rule applies even when no sentence mentions Pulmonary fibrotic sequela. If sentences deny Pulmonary fibrotic sequela, none may do so any more: a sentence that denies only that is removed, except that the first such sentence is replaced with the sentence above; a sentence that denies other things too keeps those and drops only Pulmonary fibrotic sequela. The sentence above appears exactly once in the report.
- Why: The classifier probability is direct image evidence for Pulmonary fibrotic sequela. A higher value indicates the scan more likely shows this finding, which correctly supports adding the finding to the report. The evidence is specific to the same finding, and the direction (more evidence → add) is appropriate.
- Record: {'n': 119, 'dF1': 0.24063566201193964, 'dG': -0.28657547725344357, 'precision': 0.5378151260504201}

## R12 · scoliosis · remove

- When (code): every eligible draft
- Do: The report must no longer state scoliosis. Remove that statement everywhere, also inside list sentences and in the impression. When a sentence states other findings too, keep them and drop only this one; delete a sentence only when nothing else in it remains.
- Why: not reviewed: no condition, or an edit that carries its own image evidence
- Record: {'n': 42, 'dF1': -0.010773109365132427, 'dG': 6.404543965519576, 'agree': 0.30952380952380953}

## R13 · Medical material · type

- When (code): metal_mm3 <= 33.2952
- Do: Add '<device sentence>' to the findings.
- Why: not reviewed: no condition, or an edit that carries its own image evidence
- Record: {'n': 49, 'dF1': 0.011173876551708872, 'dG': 0.5262936896228513, 'precision': 0.7346938775510204}

"""The 18 CT-RATE abnormalities, their reference labels and the macro F1 used everywhere."""
import re

import numpy as np
import pandas as pd

from rulebank.paths import CTRATE

FINDINGS = ["Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
            "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
            "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
            "Peribronchial thickening", "Consolidation", "Bronchiectasis", "Interlobular septal thickening"]   # CT-RATE order
KEYS = {"Medical material": r"stent|pacemaker|catheter|prosthe|sternotomy|wire|clip|port\b|device",
        "Arterial wall calcification": r"atheroscl|atherom|calcified plaque|aort\w* .*calcif|calcif\w* .*aort",
        "Cardiomegaly": r"cardiomegaly|heart size|cardiothoracic|enlarged heart|heart .*enlarg",
        "Pericardial effusion": r"pericardial (effusion|fluid)", "Coronary artery wall calcification": r"coronary",
        "Hiatal hernia": r"hiatal|hiatus", "Lymphadenopathy": r"lymph", "Emphysema": r"emphysema",
        "Atelectasis": r"atelecta", "Lung nodule": r"nodul", "Lung opacity": r"ground.glass|opacit|densit",
        "Pulmonary fibrotic sequela": r"fibro|sequela|scar|band", "Pleural effusion": r"pleural (effusion|fluid)",
        "Mosaic attenuation pattern": r"mosaic", "Peribronchial thickening": r"peribronch|bronchial wall thick",
        "Consolidation": r"consolidat", "Bronchiectasis": r"bronchiectas", "Interlobular septal thickening": r"septal"}
KEY = [re.compile(KEYS[f], re.I) for f in FINDINGS]             # a sentence names the abnormality
DENIAL = re.compile(r"^\s*(no|without|not|absent|negative|normal)\b|\bno (evidence|sign)|\bnot (seen|identified|observed|detected)|"
                 r"were not|was not|is not|are not|\bnormal\b", re.I)      # a sentence denies something
NEGATED = re.compile(r"\bno\b|\bnot\b|without|absence", re.I)


def ctrate_labels(ids, ct_split):
    lab = pd.read_csv(f"{CTRATE}/labels/multi_abnormality_labels/{ct_split}_predicted_labels.csv")
    lab.index = lab["VolumeName"].str.replace(".nii.gz", "", regex=False)
    return lab.loc[ids, FINDINGS].to_numpy(int)


def macro_f1(P_, Y):
    tp = (P_ & Y).sum(0)
    return 100 * np.mean(2 * tp / np.maximum(1, P_.sum(0) + Y.sum(0)))

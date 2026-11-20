"""Independent evidence per scan: classifier probabilities and image measurements, and the drafts they are read against.
The evidence comes from the tables of stage 0. Classifier probabilities: development = out-of-fold, valid = fitted on train."""
import json
import os

import numpy as np
import pandas as pd

from rulebank.labels import FINDINGS, ctrate_labels
from rulebank.paths import EVIDENCE, V5, V6

CLASSIFIERS = ("FX", "HL", "AS")                                # FlexiCT, HLIP, ASAP; the rules read their mean. Not COLIPRI: both generators' visual encoders are built on it
DEV_SPLIT = {"nv": "rot3_dev3000", "selfcue": "dev3000"}        # NV's development drafts are from CT rotated by 3 degrees
VALID_SPLIT = "valid1564"
EVIDENCE_SPLIT = {"rot3_dev3000": "dev3000", "dev3000": "dev3000", "valid1564": "valid1564"}
CT_SPLIT = {"dev3000": "train", "valid1564": "valid"}
CASES = {"dev3000": f"{V5}/inputs/cases_v1_2026-10-27/dev3000.jsonl", "valid1564": f"{V5}/inputs/cases_v1_2026-10-27/valid1564.jsonl"}
SENTENCE_RADBERT = {("nv", "valid1564"): f"{V5}/results/nv_valid_ceiling_2026-10-28/nv_valid_sentences.npz",
                    ("nv", "rot3_dev3000"): f"{V5}/results/nv_meas_sweep/sents_rot3_dev3000.npz",
                    ("selfcue", "dev3000"): f"{V5}/results/selfcue_meas_sweep/sents_dev3000.npz",
                    ("selfcue", "valid1564"): f"{V5}/results/selfcue_meas_sweep/sents_valid1564.npz"}


DRAFT_SEED = os.environ["DRAFT_SEED"] if "DRAFT_SEED" in os.environ else None    # draft-level variation (RJ 2026-11-08): the sampled drafts of this seed (stage 0c) in place of the greedy drafts


def draft_dir(gen, split):
    """where the drafts of this generator and split live"""
    return f"{V6}/results/drafts_t07/{gen}/{EVIDENCE_SPLIT[split]}_seed{DRAFT_SEED}" if DRAFT_SEED else f"{V5}/results/{gen}/{split}"


def green_references(gen, split):
    """the references GREEN reads. NV: the same sentences laid out in NV's sections, so that format does not count as an error (RJ 2026-11-06)"""
    ev = EVIDENCE_SPLIT[split]
    if gen == "nv":
        return {x["vid"]: x["text"] for x in map(json.loads, open(f"{V5}/results/green_nvref_2026-11-04/refs_{ev}.jsonl"))}
    return {x["vid"]: x["reference"] for x in map(json.loads, open(CASES[ev]))}


class DraftSet:
    """the drafts of one generator on one split, with their labels, per-sentence RadBERT and evidence"""
    def __init__(self, gen, split):
        self.gen, self.split = gen, split
        self.evidence_split = ev = EVIDENCE_SPLIT[split]
        path = draft_dir(gen, split)
        self.original = {x["vid"]: x["text"] for x in map(json.loads, open(f"{path}/drafts.jsonl"))}
        self.ids = sorted(self.original)
        z = np.load(f"{path}/radbert.npz" if DRAFT_SEED else f"{path}/metrics/draft/radbert.npz")
        D = dict(zip(z["vid"].tolist(), z["y_pred"]))
        self.labels = np.array([D[v] for v in self.ids]).astype(bool)
        self.reference_labels = ctrate_labels(self.ids, CT_SPLIT[ev]).astype(bool)
        self.references = {c["vid"]: c["reference"] for c in map(json.loads, open(CASES[ev]))}
        clf = pd.read_parquet(f"{EVIDENCE}/classifiers/{ev}.parquet").loc[self.ids]
        self.classifier_probs = {k: clf[[f"{k}:{f}" for f in FINDINGS]].to_numpy() for k in CLASSIFIERS}
        self.classifier_mean = np.mean([self.classifier_probs[k] for k in CLASSIFIERS], 0)
        self.measurements = pd.read_parquet(f"{EVIDENCE}/measurements/{ev}.parquet").loc[self.ids]
        self.ascending_mm = self.measurements["ascending_mm"].to_numpy()
        self.side = self.measurements["side"].tolist()                                   # measured pleural effusion side
        self.device_types = [set(t.split("|")) - {""} for t in self.measurements["device_types"]]
        z = np.load(f"{path}/sents.npz" if DRAFT_SEED else SENTENCE_RADBERT[(gen, split)])
        self.sentence_probs = {}
        for v, i, p in zip(z["vid"], z["id"], z["soft"]):
            self.sentence_probs.setdefault(str(v), {})[int(i)] = p

    def evidence(self, names):
        """{name: value per draft}; clf:<abnormality> = the classifier mean, any other name = a measurement"""
        return {z: self.draft_signal(z) if z.startswith("draft") else self.classifier_mean[:, FINDINGS.index(z[4:])] if z.startswith("clf:") else self.measurements[z].to_numpy(float) for z in names}

    def draft_signal(self, z):
        """a value computed from the draft's text alone (the control without independent evidence, RJ 2026-11-08): draft:<abnormality> = RadBERT on the whole draft, draft_n_abnormalities, draft_words"""
        assert DRAFT_SEED is None, "the control without independent evidence reads the greedy drafts"
        if z == "draft_n_abnormalities":
            return self.labels.sum(1).astype(float)
        if z == "draft_words":
            return np.array([len(self.original[v].split()) for v in self.ids], dtype=float)
        path = f"{EVIDENCE}/draft_radbert/selfcue_valid1564.npz" if (self.gen, self.split) == ("selfcue", "valid1564") else f"{V5}/results/{self.gen}/{self.split}/metrics/draft/radbert.npz"   # stage 0b: main_v5 kept no scores for this one
        d = np.load(path)
        by_vid = dict(zip(d["vid"].tolist(), d["soft"]))
        return np.array([by_vid[v] for v in self.ids])[:, FINDINGS.index(z[6:])].astype(float)

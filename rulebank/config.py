"""Experiment configuration under configs/: an experiment is one YAML file, validated here. Every field is required and an unknown field is an error.
The command line only says how to run it (generator, server, replay)."""
import os
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from rulebank.paths import CONFIGS


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LearnerConfig(Strict):
    prompt: Literal["explore", "explore_p2"]     # explore_p2: the same prompt, reworded
    enum_names: bool                             # finding and evidence names are enums in the JSON schema
    anon: bool                                   # ablation: anonymous evidence, no descriptions
    no_experience: bool                          # ablation: the prompt keeps only the last round's outcomes
    rounds: int
    budget: int                                  # probes per round
    blind: bool = False                          # control (tab:revision-feedback): no outcome and no bank summary in the prompt, only the probes tried
    repeat_retries: int                          # a probe that repeats an earlier one is refused and others asked for, up to this many times per round; 0 = run it


class GateConfig(Strict):
    margin: float                                # the GREEN change must be at least this
    margin_add: float                            # a labelled addition: the GREEN change must be at least this, in place of margin
    min_agree_add: float                         # out-of-label addition: share of changed drafts whose reference states it
    max_agree_remove: float                      # out-of-label removal: the same share, at most this
    min_changed_per_3000: int                    # a rule must change at least this many drafts, scaled to the drafts it may see
    min_precision_add: float                     # a labelled addition: share of changed drafts whose reference label is positive, at least this


class FoldsConfig(Strict):
    outer: int                                   # outer folds by patient, for the held-out estimate
    seed: int


class LLMConfig(Strict):
    model: str
    temperature: float
    reasoning_effort: str
    max_tokens: int
    seed: int | None = None                      # sampling seed passed to the server (tab:learning-variation); None = not sent


class Experiment(Strict):
    name: str                                    # the file name, and the output directory <out-root>/<name>/<gen>
    table: str                                   # the output directory of stage 1
    evidence: Literal["main", "wide", "draft_only", "clf_only", "meas_only"]   # configs/evidence/<name>.yaml
    learner: LearnerConfig
    gate: GateConfig
    folds: FoldsConfig
    llm: LLMConfig


class EditorConfig(Strict):
    model: str
    temperature: float
    retry_temperature: float
    retries: int                                 # further tries when a reply is unusable
    reasoning_effort: str
    max_tokens: int
    workers: int                                 # parallel calls
    seed: int | None = None                      # read by llm.Caller like LLMConfig.seed; None = not sent (every editor so far). Added 2026-11-10: without it every editor call failed after the seed was added to llm.py


def load_editor(path):
    return EditorConfig(**yaml.safe_load(open(path)))


def load_experiment(path):
    cfg = Experiment(**yaml.safe_load(open(path)))
    assert cfg.name == os.path.basename(path).removesuffix(".yaml"), (cfg.name, path)
    return cfg


def evidence_names(name):
    """the evidence of the set configs/evidence/<name>.yaml"""
    return yaml.safe_load(open(f"{CONFIGS}/evidence/{name}.yaml"))["evidence"]


def edits_off(name):
    """the edits the set configs/evidence/<name>.yaml cannot support: their written content comes from the image"""
    return yaml.safe_load(open(f"{CONFIGS}/evidence/{name}.yaml"))["edits_off"]

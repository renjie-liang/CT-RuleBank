"""Stage 2, GPU, gpt-oss-120b (the launcher appends --base-url). The rule agent learns a bank: one run per outer fold for the held-out estimate,
one on all development drafts for the bank that is applied. The experiment is one file in configs/experiments/.
    MODULE=steps.stage2_learn_bank ARGS="--config configs/experiments/v9_m0.yaml --gen selfcue --out-root results" sbatch slurm/gptoss_run_compact.sbatch"""
import argparse

from rulebank import learner
from rulebank.config import load_experiment

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--gen", required=True, choices=["nv", "selfcue"])
ap.add_argument("--out-root", required=True, help="the run goes to <out-root>/<experiment name>/<gen>")
ap.add_argument("--folds", default="0,1,2,3,4,final")
ap.add_argument("--base-url", help="the local vLLM server")
ap.add_argument("--exchange", help="answer through files (rulebank.file_exchange): Claude via Claude Code subagents")
ap.add_argument("--random-from", help="no LLM: random probes, as many per fold as the agent run in this directory evaluated (tab:rule-search-control)")
ap.add_argument("--replay", help="a recorded trace.jsonl: answer every LLM call from it, prompts asserted equal")
a = ap.parse_args()
learner.main(a.gen, load_experiment(a.config), a.out_root, a.folds.split(","), a.base_url, a.replay, a.exchange, a.random_from)

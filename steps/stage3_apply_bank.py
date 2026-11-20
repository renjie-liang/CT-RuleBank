"""Stage 3. Learned banks on the validation drafts: the revised reports, and the inputs of CT-FineBench. The learning simulates the editor by code edits; the final test uses the editor agent.
    python -m steps.stage3_apply_bank --gen selfcue --runs mybank=results/learn --out results/valid --executor code                                     (CPU)
    MODULE=steps.stage3_apply_bank ARGS="--gen selfcue --runs mybank=results/learn --out results/valid --executor llm --editor configs/editor/gptoss.yaml" sbatch slurm/gptoss_run_compact.sbatch"""
import argparse

from baselines import text_memory
from rulebank import apply, editor
from rulebank.config import load_editor

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True, choices=["nv", "selfcue"])
ap.add_argument("--runs", nargs="+", required=True, help="name=<output directory of stage 2>, one per bank; name=<dir>@<gen> takes the bank learned for that other generator")
ap.add_argument("--out", required=True)
ap.add_argument("--executor", required=True, choices=["code", "llm", "text_memory"], help="code edits, the editor agent (LLM + code), or the text memory control (LLM decides which rules apply)")
ap.add_argument("--editor", help="llm: the editor's config file")
ap.add_argument("--base-url", help="llm: the local vLLM server")
ap.add_argument("--exchange", help="llm: answer through files (rulebank.file_exchange): Claude via Claude Code subagents")
ap.add_argument("--limit", type=int, default=0, help="llm: the first N drafts with a rule that fired, for a smoke test")
a = ap.parse_args()
apply.collect_banks(a.gen, a.runs, a.out)
if a.executor == "code":
    apply.revise_drafts(a.gen, a.out)
elif a.executor == "text_memory":
    text_memory.edit_drafts(a.gen, a.out, load_editor(a.editor), a.base_url, a.limit)
else:
    editor.edit_drafts(a.gen, a.out, load_editor(a.editor), a.base_url, a.limit, a.exchange)
apply.write_finebench_inputs(a.gen, a.out)

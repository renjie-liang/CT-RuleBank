"""The edit-outcome table: for every development draft and every edit the library can make, the revised text, its RadBERT labels and its
paired GREEN change. A rule only decides which drafts get an edit, so every rule is scored by lookup, with no new text or GPU.
Built by stage 1 -> <out>/<gen>/: table.jsonl, counts.md, green_work.jsonl, radbert.npz, green/shard*.jsonl."""
import collections
import glob
import json
import os

import numpy as np

from rulebank.edits import possible_edits
from rulebank.evidence import DEV_SPLIT, DraftSet, green_references
from rulebank.gate import base_edit


def build_texts(gen, out, limit):
    d = os.path.join(out, gen)
    os.makedirs(d, exist_ok=False)
    split = DEV_SPLIT[gen]
    drafts = DraftSet(gen, split)
    ks = range(len(drafts.ids)) if limit == 0 else range(limit)
    n = collections.Counter()
    with open(os.path.join(d, "table.jsonl"), "w") as f:
        for k in ks:
            v = drafts.ids[k]
            for fnd, e, t in possible_edits(drafts, k, drafts.original[v], drafts.sentence_probs[v]):
                n[(fnd, e, "text" if t is not None else "abstain")] += 1
                if t is not None:
                    f.write(json.dumps({"vid": v, "finding": fnd, "edit": e, "text": t}) + "\n")
    L = [f"# Edit-outcome table, {gen} {split} ({len(ks)} drafts)", "", "| abnormality | edit | texts | abstain |", "|---|---|---|---|"]
    for fnd, e in sorted({(a, base_edit(b)) for a, b, _ in n}):
        tx = sum(v for (a, b, s), v in n.items() if a == fnd and base_edit(b) == e and s == "text")
        ab = sum(v for (a, b, s), v in n.items() if a == fnd and base_edit(b) == e and s == "abstain")
        L.append(f"| {fnd} | {e} | {tx} | {ab} |")
    L += ["", f"total texts {sum(v for (a, b, s), v in n.items() if s == 'text')}"]
    open(os.path.join(d, "counts.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    write_green_work(d, drafts)


def write_green_work(d, drafts):
    """the GREEN work list: every draft with a table row once, then every table row, so drafts and edits share the same GREEN runs"""
    rows = [json.loads(l) for l in open(os.path.join(d, "table.jsonl"))]
    work = [{"vid": v, "kind": "draft", "row": -1, "text": drafts.original[v]} for v in sorted({r["vid"] for r in rows})]
    work += [{"vid": r["vid"], "kind": "edit", "row": k, "text": r["text"]} for k, r in enumerate(rows)]
    with open(os.path.join(d, "green_work.jsonl"), "x") as f:
        for i, w in enumerate(work):
            f.write(json.dumps({"i": i, **w}) + "\n")
    print(f"green work list: {len(work)} items ({sum(w['kind'] == 'draft' for w in work)} drafts)")


def score_radbert(gen, out):
    from eval.radbert import radbert_soft
    rows = [json.loads(l) for l in open(os.path.join(out, gen, "table.jsonl"))]
    uniq = sorted({r["text"] for r in rows})
    P = dict(zip(uniq, radbert_soft(uniq, device="cuda", bs=32)))
    np.savez(os.path.join(out, gen, "radbert.npz"), row=np.arange(len(rows)), soft=np.stack([P[r["text"]] for r in rows]).astype(np.float16))
    print(f"[radbert] {gen}: {len(rows)} rows, {len(uniq)} distinct texts", flush=True)


def score_green(gen, out, shard, nshards):
    from eval.green import score
    refs = green_references(gen, DEV_SPLIT[gen])
    task = [w for w in map(json.loads, open(os.path.join(out, gen, "green_work.jsonl"))) if w["i"] % nshards == shard]
    os.makedirs(os.path.join(out, gen, "green"), exist_ok=True)
    res = score([refs[w["vid"]] for w in task], [w["text"] for w in task])
    with open(os.path.join(out, gen, "green", f"shard{shard:03d}.jsonl"), "x") as f:
        for w, (g, an) in zip(task, res):
            f.write(json.dumps({"i": w["i"], "green": g, "analysis": an}) + "\n")
    print(f"[green] {gen} shard {shard}/{nshards}: {len(task)} items", flush=True)


def usable(g):
    """the GREEN answer's score, unless it did not parse or claims a report was 'not provided' (a failure of the judge)"""
    return g["green"] if g["green"] is not None and "not provided" not in g["analysis"] else None


class EditTable:
    """the table loaded for lookup: rows, RadBERT labels, paired GREEN change per row in points"""
    def __init__(self, table, gen, ids):
        d = os.path.join(table, gen)
        self.rows = [json.loads(l) for l in open(os.path.join(d, "table.jsonl"))]
        self.labels = np.load(os.path.join(d, "radbert.npz"))["soft"] >= 0.5
        task = {w["i"]: w for w in map(json.loads, open(os.path.join(d, "green_work.jsonl")))}
        G = {g["i"]: g for f in glob.glob(os.path.join(d, "green", "shard*.jsonl")) for g in map(json.loads, open(f))}
        assert set(G) == set(task), f"GREEN incomplete: {len(G)} of {len(task)}"
        gd = {task[i]["vid"]: usable(G[i]) for i in task if task[i]["kind"] == "draft"}
        ge = {task[i]["row"]: usable(G[i]) for i in task if task[i]["kind"] == "edit"}
        self.dgreen = np.array([np.nan if ge[k] is None or gd[r["vid"]] is None else 100 * (ge[k] - gd[r["vid"]]) for k, r in enumerate(self.rows)])
        self.position = {v: k for k, v in enumerate(ids)}
        self.eligible_rows = collections.defaultdict(list)             # (abnormality, edit) -> rows of the drafts eligible for it
        for k, r in enumerate(self.rows):
            self.eligible_rows[(r["finding"], base_edit(r["edit"]))].append(k)

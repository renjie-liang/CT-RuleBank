"""Evaluation of a learned bank on development drafts it did not learn from: outer folds by patient, joint label lookup."""
import numpy as np

from rulebank.labels import FINDINGS, macro_f1


def patient_of(v):
    return "_".join(v.split("_")[:2])


def patient_folds(V, k, seed):
    up = np.unique([patient_of(v) for v in V])
    up = np.random.default_rng(seed).permutation(up)
    fold_of = {p: n % k for n, p in enumerate(up)}
    return [{v for v in V if fold_of[patient_of(v)] == f} for f in range(k)]


def patient_bootstraps(ids, n, rng):
    pid = np.array([patient_of(v) for v in ids])
    up = np.unique(pid)
    return [np.concatenate([np.where(pid == q)[0] for q in rng.choice(up, len(up))]) for _ in range(n)]


def fired_rows(table, rules, V):
    """{(draft, abnormality): table row} of rules {(abnormality, edit): rule} on drafts V; add_typed before add for the same pair"""
    out = {}
    for (f, e), r in sorted(rules.items(), key=lambda kv: kv[0][1] != "add_typed"):
        for k in r["rows"]:
            v = table.rows[k]["vid"]
            if v in V and (v, f) not in out:
                out[(v, f)] = k
    return out


def evaluate_edits(F, table, drafts, B):
    """F = fired_rows(...) over all drafts: macro F1 change by label lookup, patient-bootstrap CI; mean paired GREEN change over the fired rows"""
    P = drafts.labels.copy()
    for (v, f), k in F.items():
        if f in FINDINGS:
            j = FINDINGS.index(f)
            P[table.position[v], j] = table.labels[k][j]
    d = macro_f1(P, drafts.reference_labels) - macro_f1(drafts.labels, drafts.reference_labels)
    bs = np.array([macro_f1(P[b], drafts.reference_labels[b]) - macro_f1(drafts.labels[b], drafts.reference_labels[b]) for b in B])
    g = table.dgreen[list(F.values())]
    g = g[np.isfinite(g)]
    rng = np.random.default_rng(1)
    gb = np.array([rng.choice(g, len(g)).mean() for _ in range(1000)]) if len(g) else np.array([np.nan])
    return {"dF1": float(d), "dF1_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "dG": float(g.mean()) if len(g) else None, "dG_ci": [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))],
            "edits": len(F), "drafts": len({v for v, f in F})}

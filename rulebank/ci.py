"""The one confidence interval of the paper (RJ 2026-11-09): patients resampled with replacement, 2,000 resamples, a fixed seed, the 2.5 and 97.5 percentiles. Every validation result uses it: summary.md, absolute.json, changed_subset.json, curve.md."""
import numpy as np

from rulebank import evaluate as E
from rulebank.labels import macro_f1

B = 2000
SEED = 0


def resamples(ids):
    """index arrays into ids, one per resample"""
    return E.patient_bootstraps(list(ids), B, np.random.default_rng(SEED))


def interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def mean_interval(x, ids):
    """the 95% interval of the mean of x (one value per report in ids)"""
    x = np.asarray(x, dtype=float)
    return interval([x[b].mean() for b in resamples(ids)])


def detection(D, P, Y, ids):
    """macro F1 and CRG of drafts D and revised P against Y over the reports ids -> {f1: {draft, revised, diff}, crg: ...}, each a 95% interval"""
    from eval.crg import crg
    b = resamples(ids)
    fd, fp = np.array([macro_f1(D[x], Y[x]) for x in b]), np.array([macro_f1(P[x], Y[x]) for x in b])
    cd, cp = np.array([100 * crg(D[x], Y[x]) for x in b]), np.array([100 * crg(P[x], Y[x]) for x in b])
    return {"f1": {"draft": interval(fd), "revised": interval(fp), "diff": interval(fp - fd)}, "crg": {"draft": interval(cd), "revised": interval(cp), "diff": interval(cp - cd)}}

"""Small, dependency-light statistics helpers shared by the iter1 readout /
report scripts (analyze_token_signals.py, fcvr_input_level_ood_check.py,
abstention_report.py, baseline_ladder_report.py).

Everything here is example-level: bootstrap resampling is over EXAMPLES (not
tokens), because tokens within an example are autocorrelated.
"""
import numpy as np


def auroc(labels, scores):
    """AUROC with the SIGN FIXED A PRIORI: higher score => positive class (wrong / OoD).
    Never flip. Returns nan if one class is missing."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    ok = ~np.isnan(scores)
    labels, scores = labels[ok], scores[ok]
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        # Mann-Whitney fallback with tie handling
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        s_sorted = scores[order]
        i = 0
        while i < len(s_sorted):
            j = i
            while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        n_pos = labels.sum(); n_neg = labels.size - n_pos
        return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(labels, scores):
    labels = np.asarray(labels, dtype=int); scores = np.asarray(scores, dtype=float)
    ok = ~np.isnan(scores); labels, scores = labels[ok], scores[ok]
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, scores))
    except Exception:
        return float("nan")


def bootstrap_ci(stat_fn, labels, scores, n_boot=2000, seed=0, alpha=0.05, stratified=True):
    """Example-level (optionally class-stratified) percentile bootstrap CI of
    stat_fn(labels, scores). Returns dict(point, lo, hi, n_boot, n)."""
    labels = np.asarray(labels, dtype=int); scores = np.asarray(scores, dtype=float)
    n = labels.size
    point = stat_fn(labels, scores)
    if n < 3 or np.isnan(point):
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": 0, "n": int(n)}
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1); neg = np.flatnonzero(labels == 0)
    vals = []
    for _ in range(n_boot):
        if stratified and pos.size and neg.size:
            idx = np.concatenate([rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)])
        else:
            idx = rng.integers(0, n, n)
        v = stat_fn(labels[idx], scores[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": 0, "n": int(n)}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": len(vals), "n": int(n)}


def bootstrap_mean_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean of per-example values (e.g. per-example Spearman)."""
    v = np.asarray([x for x in values if not (x is None or np.isnan(x))], dtype=float)
    if v.size < 3:
        return {"mean": float(v.mean()) if v.size else float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(v.size)}
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size), "std": float(v.std())}


def risk_coverage(labels_wrong, scores, coverages=(1.0, 0.9, 0.8, 0.7, 0.5)):
    """Selective prediction: keep the (1-c) least-uncertain examples... i.e. at
    coverage c, ABSTAIN on the top (1-c) fraction by score (higher = more
    uncertain). Returns per-coverage risk (error rate among kept), the score
    threshold that achieves it, and AURC (area under the risk-coverage curve
    over all coverages, lower = better)."""
    y = np.asarray(labels_wrong, dtype=int); s = np.asarray(scores, dtype=float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    n = y.size
    if n == 0:
        return {"aurc": float("nan"), "points": {}}
    order = np.argsort(s)  # ascending uncertainty: kept first
    y_sorted = y[order]
    cum_err = np.cumsum(y_sorted) / np.arange(1, n + 1)   # risk at coverage k/n
    aurc = float(np.mean(cum_err))
    points = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        thr = float(s[order][k - 1])
        points[str(c)] = {"coverage": k / n, "risk": float(cum_err[k - 1]), "threshold": thr, "n_kept": int(k)}
    return {"aurc": aurc, "points": points, "n": int(n)}


def apply_thresholds(labels_wrong, scores, thresholds):
    """Evaluate FROZEN thresholds (from val) on a new set: keep examples with
    score <= thr. Returns coverage / risk / accuracy-of-kept per threshold."""
    y = np.asarray(labels_wrong, dtype=int); s = np.asarray(scores, dtype=float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    out = {}
    for name, thr in thresholds.items():
        keep = s <= thr
        k = int(keep.sum())
        out[name] = {"threshold": float(thr), "coverage": k / max(1, y.size),
                     "risk": float(y[keep].mean()) if k else float("nan"),
                     "accuracy_kept": float(1 - y[keep].mean()) if k else float("nan"), "n_kept": k}
    return out


def calibration_metrics(labels_wrong, probs, n_bins=10):
    """Brier, NLL, ECE (equal-width bins) and adaptive ECE (equal-mass bins) of
    predicted P(wrong)."""
    y = np.asarray(labels_wrong, dtype=float); p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def ece(bins_edges):
        e = 0.0
        for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
            m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
            if m.any():
                e += m.mean() * abs(p[m].mean() - y[m].mean())
        return float(e)
    ece_eq = ece(np.linspace(0, 1, n_bins + 1))
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1)); qs[0], qs[-1] = 0.0, 1.0
    ece_ad = ece(np.unique(qs))
    return {"brier": brier, "nll": nll, "ece": ece_eq, "ece_adaptive": ece_ad, "n": int(y.size)}

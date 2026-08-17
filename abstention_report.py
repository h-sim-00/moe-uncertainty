"""Abstention readout with a frozen val -> test protocol (supervisor points 6, 8).

Two phases, two files, never mixed:

  --select   on the VAL labelled seqlevel file (50 ex). Records, per score, the
             AUROC(wrong) with CI, checks the a-priori sign (higher => wrong;
             a val AUROC < 0.5 is recorded as "val disagrees" -- the sign is
             NOT flipped), and freezes abstention thresholds at the target
             coverages. Also fits the information-beyond-baselines logistic
             models (baselines vs baselines+ILV) on val. Writes
             results/abstention/frozen_<tag>.json (with a config hash).
  --evaluate on the TEST labelled seqlevel file (175 ex), ONCE per frozen file:
             AUROC + example-level bootstrap CI per score, risk/coverage/accuracy
             at the FROZEN thresholds, AURC, the val-fitted logistic models
             applied unchanged, and a likelihood-ratio test of +ILV on test
             (reported as analysis; no decision is derived from it). Refuses to
             run without a frozen file; warns if the input is not a test file.
  --aggregate eval_*.json ...  -> mean +- sd across inference seeds.

Label: --label correct_primary (pre-registered: option_correct with explicit
probe fallback); alternatives (correct_probe, expl_entails, option_correct) are
reported as secondary via --extra_labels.
Primary score: --primary_score ilv_online_mean_last10 (pre-registered);
every ilv_* / *_BASELINE score in the file is reported.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

from uq_stats import auroc, bootstrap_ci, risk_coverage, apply_thresholds

BASELINE_KEYS = ["entropy_mean_BASELINE", "entropy_max_BASELINE", "nll_per_token_BASELINE", "gen_len_BASELINE"]


# ---------------------------------------------------------------------------
def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def labelled(rows, label):
    keep = [r for r in rows if r.get(label) is not None]
    y = np.array([0 if r[label] else 1 for r in keep], dtype=int)      # 1 = WRONG
    return keep, y


def score_matrix(rows, names):
    return np.array([[r["scores"].get(n, np.nan) for n in names] for r in rows], dtype=float)


def all_score_names(rows):
    names = sorted({k for r in rows for k in r.get("scores", {})})
    return [n for n in names if all(n in r["scores"] for r in rows)]


def config_hash(labeled_path):
    """Hash of the readout config (arm, routing, S, run_suffix, prior, max_new_tokens)
    so a frozen file can only be applied to a test run of the SAME configuration."""
    stem = labeled_path.replace("_seqlevel_labeled.jsonl", "")
    stem = stem.replace("_val_", "_{split}_").replace("step1_medexqa_generate", "step1_medexqa_{split}_generate")
    cfg = {}
    for cand in (labeled_path.replace("_seqlevel_labeled.jsonl", ".json"),):
        if os.path.exists(cand):
            cfg = json.load(open(cand)).get("config", {})
    keys = ("arm", "routing", "num_samples", "run_suffix", "prior_source", "map_suffix", "max_new_tokens", "fcvr_layers")
    payload = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12], {k: cfg.get(k) for k in keys}


# ---------------------------------------------------------------------------
# tiny logistic regression (L2, IRLS) so the residual test has no sklearn dependency
def fit_logreg(X, y, l2=1e-2, iters=100):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H + 1e-8 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def logreg_predict(w, X):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return 1 / (1 + np.exp(-Xb @ w))


def loglik(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def standardise(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


# ---------------------------------------------------------------------------
def do_select(args):
    rows = load_rows(args.input)
    if "_val_" not in os.path.basename(args.input) and not args.force:
        sys.exit("--select expects a VAL file (name contains '_val_'); pass --force to override (and say why in the notes).")
    keep, y = labelled(rows, args.label)
    names = all_score_names(keep)
    if args.primary_score not in names:
        sys.exit(f"primary score {args.primary_score} not present on every row; available: {names}")
    ch, cfg = config_hash(args.input)
    frozen = {
        "tag": args.tag, "label": args.label, "primary_score": args.primary_score,
        "sign": "a priori: higher score => WRONG. Never flipped.",
        "coverages": args.coverages, "n_val": int(len(keep)), "n_wrong_val": int(y.sum()),
        "config_hash": ch, "config": cfg, "val_file": args.input,
        "scores": {}, "thresholds": {}, "residual_models": {},
    }
    S = score_matrix(keep, names)
    for j, n in enumerate(names):
        ci = bootstrap_ci(auroc, y, S[:, j], n_boot=args.n_boot)
        rc = risk_coverage(y, S[:, j], coverages=args.coverages)
        frozen["scores"][n] = {"val_auroc": ci["point"], "val_auroc_ci95": [ci["lo"], ci["hi"]],
                               "val_disagrees_with_a_priori_sign": bool(ci["point"] < 0.5),
                               "val_aurc": rc["aurc"]}
        frozen["thresholds"][n] = {str(c): rc["points"][str(c)]["threshold"] for c in args.coverages}
    # information-beyond-baselines: logistic models fitted on val (applied unchanged on test)
    base = [b for b in BASELINE_KEYS if b in names]
    Xb = score_matrix(keep, base)
    Xb_s, mu_b, sd_b = standardise(Xb)
    w_b = fit_logreg(Xb_s, y)
    ilv_scores = [n for n in names if n.startswith("ilv_")]
    frozen["residual_models"] = {"baselines": base, "baseline_w": w_b.tolist(), "baseline_mu": mu_b.tolist(),
                                 "baseline_sd": sd_b.tolist(), "with_ilv": {}}
    for n in ilv_scores:
        X = np.hstack([Xb, S[:, [names.index(n)]]])
        Xs, mu, sd = standardise(X)
        w = fit_logreg(Xs, y)
        frozen["residual_models"]["with_ilv"][n] = {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
                                                     "val_auroc_combo": auroc(y, logreg_predict(w, Xs)),
                                                     "val_loglik_gain": loglik(y, logreg_predict(w, Xs)) - loglik(y, logreg_predict(w_b, Xb_s))}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"frozen_{args.tag}.json")
    if os.path.exists(out) and not args.overwrite:
        sys.exit(f"{out} exists -- a frozen selection must not be silently replaced (pass --overwrite and record why).")
    json.dump(frozen, open(out, "w"), indent=2)
    p = frozen["scores"][args.primary_score]
    print(f"[SELECT/val n={len(keep)} wrong={y.sum()}] primary {args.primary_score}: val AUROC {p['val_auroc']:.3f} "
          f"[{p['val_auroc_ci95'][0]:.2f},{p['val_auroc_ci95'][1]:.2f}]"
          + ("  ** val disagrees with the a-priori sign -> report as negative; NOT flipped **" if p["val_disagrees_with_a_priori_sign"] else ""))
    print("thresholds (score <= thr is KEPT):", json.dumps(frozen["thresholds"][args.primary_score]))
    print(f"Frozen -> {out}")


def do_evaluate(args):
    if not args.frozen or not os.path.exists(args.frozen):
        sys.exit("--evaluate requires --frozen results/abstention/frozen_<tag>.json produced by --select on VAL.")
    frozen = json.load(open(args.frozen))
    if "_val_" in os.path.basename(args.input) and not args.force:
        print("WARNING: evaluating on a VAL file (sanity only). The protocol result is the TEST run.")
    rows = load_rows(args.input)
    ch, cfg = config_hash(args.input)
    if ch != frozen["config_hash"] and not args.force:
        sys.exit(f"config hash mismatch: frozen {frozen['config_hash']} {frozen['config']} vs test {ch} {cfg}. "
                 f"The frozen selection may only be applied to the SAME configuration (--force to override).")
    label = frozen["label"]
    keep, y = labelled(rows, label)
    names = [n for n in all_score_names(keep) if n in frozen["scores"]]
    S = score_matrix(keep, names)
    res = {"tag": frozen["tag"], "frozen": args.frozen, "test_file": args.input, "label": label,
           "primary_score": frozen["primary_score"], "n_test": int(len(keep)), "n_wrong_test": int(y.sum()),
           "accuracy_test": float(1 - y.mean()) if len(y) else None, "scores": {}, "residual": {}}
    for j, n in enumerate(names):
        ci = bootstrap_ci(auroc, y, S[:, j], n_boot=args.n_boot)
        rc = risk_coverage(y, S[:, j], coverages=frozen["coverages"])
        thr = {f"cov{c}": t for c, t in frozen["thresholds"][n].items()}
        res["scores"][n] = {"auroc": ci["point"], "auroc_ci95": [ci["lo"], ci["hi"]],
                            "ci_excludes_0.5": bool(ci["lo"] > 0.5 or ci["hi"] < 0.5),
                            "aurc": rc["aurc"], "frozen_thresholds": apply_thresholds(y, S[:, j], thr),
                            "val_auroc": frozen["scores"][n]["val_auroc"]}
    # residual test: val-fitted models applied unchanged + LR test refit on test (analysis only)
    rm = frozen["residual_models"]
    base = [b for b in rm["baselines"] if b in names]
    if base and len(base) == len(rm["baselines"]):
        Xb = score_matrix(keep, base)
        Xb_s = (Xb - np.array(rm["baseline_mu"])) / np.array(rm["baseline_sd"])
        p_b = logreg_predict(np.array(rm["baseline_w"]), Xb_s)
        res["residual"]["baselines_only"] = {"features": base, "auroc": auroc(y, p_b), "loglik": loglik(y, p_b)}
        for n, m in rm["with_ilv"].items():
            if n not in names:
                continue
            X = np.hstack([Xb, S[:, [names.index(n)]]])
            Xs = (X - np.array(m["mu"])) / np.array(m["sd"])
            p = logreg_predict(np.array(m["w"]), Xs)
            # LR test refit ON TEST (analysis; no decision derived): 2*(ll_full - ll_base) ~ chi2(1)
            Xs_t, mu_t, sd_t = standardise(X); Xb_t, mub, sdb = standardise(Xb)
            ll_full = loglik(y, logreg_predict(fit_logreg(Xs_t, y), Xs_t)); ll_base = loglik(y, logreg_predict(fit_logreg(Xb_t, y), Xb_t))
            lr = 2 * (ll_full - ll_base)
            try:
                from scipy.stats import chi2
                p_lr = float(chi2.sf(lr, df=1))
            except Exception:
                p_lr = None
            res["residual"][n] = {"auroc_val_fitted_combo": auroc(y, p), "loglik_val_fitted_combo": loglik(y, p),
                                  "loglik_gain_vs_baselines_val_fitted": loglik(y, p) - loglik(y, p_b),
                                  "lr_test_on_test": {"statistic": lr, "p_value": p_lr, "note": "analysis only; refit on test; no decision derived"}}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"eval_{frozen['tag']}{'_' + args.eval_tag if args.eval_tag else ''}.json")
    if os.path.exists(out) and not args.overwrite:
        sys.exit(f"{out} exists -- the test set is evaluated ONCE per frozen configuration (pass --overwrite only to redo a botched run).")
    json.dump(res, open(out, "w"), indent=2)
    # markdown
    lines = [f"# Abstention eval — {frozen['tag']} (label={label}, n={len(keep)}, wrong={int(y.sum())}, acc={res['accuracy_test']:.3f})",
             "", "Sign fixed a priori (higher ⇒ wrong). CI = 95% example-level bootstrap. Thresholds frozen on val.", "",
             "| score | test AUROC [CI] | val AUROC | AURC | " + " | ".join(f"risk@cov{c} (cov)" for c in frozen["coverages"]) + " |",
             "|---|---|---|---|" + "---|" * len(frozen["coverages"])]
    order = [frozen["primary_score"]] + [n for n in names if n != frozen["primary_score"]]
    for n in order:
        r = res["scores"][n]
        cells = []
        for c in frozen["coverages"]:
            t = r["frozen_thresholds"][f"cov{c}"]
            cells.append(f"{t['risk']:.3f} ({t['coverage']:.2f})")
        mark = " **(primary)**" if n == frozen["primary_score"] else ""
        lines.append(f"| {n}{mark} | {r['auroc']:.3f} [{r['auroc_ci95'][0]:.2f},{r['auroc_ci95'][1]:.2f}] | {r['val_auroc']:.3f} | {r['aurc']:.3f} | " + " | ".join(cells) + " |")
    if res["residual"]:
        lines += ["", "## Information beyond baselines (val-fitted logistic models applied unchanged)", "",
                  f"baselines only ({', '.join(res['residual'].get('baselines_only', {}).get('features', []))}): "
                  f"AUROC {res['residual'].get('baselines_only', {}).get('auroc', float('nan')):.3f}", ""]
        for n, r in res["residual"].items():
            if n == "baselines_only":
                continue
            lr = r["lr_test_on_test"]
            lines.append(f"- +{n}: combo AUROC {r['auroc_val_fitted_combo']:.3f}, Δloglik {r['loglik_gain_vs_baselines_val_fitted']:+.2f}; "
                         f"LR test on test χ²={lr['statistic']:.2f}, p={lr['p_value'] if lr['p_value'] is None else round(lr['p_value'], 4)} (analysis only)")
    md = "\n".join(lines) + "\n"
    open(out.replace(".json", ".md"), "w").write(md)
    print(md)
    print(f"Saved: {out} / .md")


def do_aggregate(args):
    files = [f for pat in args.aggregate for f in sorted(glob.glob(pat))]
    if not files:
        sys.exit("no eval files matched")
    evs = [json.load(open(f)) for f in files]
    names = sorted(set.intersection(*[set(e["scores"]) for e in evs]))
    print(f"# Aggregate over {len(evs)} runs: " + ", ".join(os.path.basename(f) for f in files))
    print("| score | test AUROC mean ± sd | min | max |"); print("|---|---|---|---|")
    agg = {}
    for n in names:
        v = np.array([e["scores"][n]["auroc"] for e in evs], float)
        agg[n] = {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max()), "n": len(v)}
        print(f"| {n} | {v.mean():.3f} ± {v.std():.3f} | {v.min():.3f} | {v.max():.3f} |")
    o = os.path.join(args.out_dir, f"aggregate_{args.tag or 'runs'}.json")
    json.dump({"files": files, "scores": agg}, open(o, "w"), indent=2)
    print(f"Saved: {o}")


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select", action="store_true", help="Fit/freeze on the VAL file.")
    mode.add_argument("--evaluate", action="store_true", help="Evaluate the TEST file once with a frozen selection.")
    mode.add_argument("--aggregate", nargs="+", default=None, help="eval_*.json globs -> mean ± sd across seeds.")
    p.add_argument("--input", help="Labelled seqlevel jsonl (val for --select, test for --evaluate/--calibrate).")
    p.add_argument("--frozen", help="frozen_<tag>.json from --select.")
    p.add_argument("--tag", default=None)
    p.add_argument("--eval_tag", default=None, help="Extra tag for the eval/calib output (e.g. seed).")
    p.add_argument("--label", default="correct_primary")
    p.add_argument("--primary_score", default="ilv_online_mean_last10")
    p.add_argument("--coverages", nargs="+", type=float, default=[0.9, 0.8, 0.7])
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out_dir", default="results/abstention")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force", action="store_true", help="Override protocol guards (records exploratory).")
    args = p.parse_args()
    if args.select:
        if not args.input or not args.tag:
            sys.exit("--select needs --input <val labelled jsonl> and --tag")
        do_select(args)
    elif args.evaluate:
        if not args.input:
            sys.exit("--evaluate needs --input <test labelled jsonl> and --frozen")
        do_evaluate(args)
    else:
        do_aggregate(args)


if __name__ == "__main__":
    main()

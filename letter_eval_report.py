"""Comparison table for the MedMCQA answer-only vs answer+explanation study
(branch MedMCQA-comparison). Collects every evaluate_letter.py output for one
split and writes one report:

  1. per-row table: one row per evaluate_letter.py --tag (zero-shot, OBQA
     reference, arm A / arm B x {kvq, det, fcvr} x seeds), fixed order
     cols = n, ACC, NLL, ECE, MCE, AUROC(wrong) [CI] per signal
  2. arm B - arm A, PAIRED: an arm-A tag is paired only with the arm-B tag whose
     remainder (method, beta, S, seed) is IDENTICAL; the delta of every metric
     (ACC, NLL, ECE, MCE, AUROC per signal) gets a paired example-level
     bootstrap CI (same resampled examples in both arms, joined on example id)
  3. multi-seed summary: tags that differ only in the read-out seed (-s<seed>)
     are pooled -> mean +- sd of every metric, and mean +- sd of the paired
     deltas across seeds
  4. per-subject ACC / ECE (medmcqa_gen only)

Tag grammar (set by run-overnight-medmcqa-arms.sh):
    <armA-letter|armB-ansexp>_<kvq|det|fcvr[-layers-<set>][-beta<b>]>[_S<S>]-s<seed>
    (the optional -layers-<set> marks a non-default FCVR layer set, OBQA-qwen; arms
    are still paired on the exact remainder, so each layer set pairs with itself)
Usage (CPU):
    python letter_eval_report.py --split test
Writes <out_dir>/letter_arms_<split>.md and .json; never touches the inputs.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from uq_stats import auroc

try:                                             # single source of truth when the full env is present
    from utils.data import TARGET_MODE_SUFFIX
except ImportError:                              # CPU-only env without torch/wandb: same values
    TARGET_MODE_SUFFIX = {"letter": "armA-letter", "answer_explanation": "armB-ansexp"}

ARM_A = TARGET_MODE_SUFFIX["letter"]             # armA-letter
ARM_B = TARGET_MODE_SUFFIX["answer_explanation"]  # armB-ansexp
SIGNALS = ["letter_entropy", "one_minus_maxprob", "gate_entropy_last", "gate_entropy_last_fcvr", "ilv_last"]
SCALARS = ["ACC", "NLL", "ECE", "MCE"]
ROW_ORDER = ["zero-shot", "ref-obqa", f"{ARM_A}_kvq", f"{ARM_A}_det", f"{ARM_A}_fcvr",
             f"{ARM_B}_kvq", f"{ARM_B}_det", f"{ARM_B}_fcvr"]
SEED_RE = re.compile(r"-s\d+$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--in_dir", type=str, default="results/letter_eval")
    p.add_argument("--out_dir", type=str, default="results/reports")
    p.add_argument("--tags", type=str, nargs="*", default=None, help="Restrict to these tags (default: all found).")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--boot_seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# metrics from per-example rows (numpy, no torch)
# ---------------------------------------------------------------------------
def ece_mce(probs, labels, n_bins=10):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); acc = (pred == labels)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0; mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(conf[m].mean() - acc[m].mean()); ece += m.mean() * gap; mce = max(mce, gap)
    return float(ece), float(mce)


def metrics_from_rows(rows, idx=None):
    """-> dict of ACC/NLL/ECE/MCE + auroc_<signal> (nan if the signal is absent)."""
    if idx is None:
        idx = np.arange(len(rows))
    probs = np.array([rows[i]["probs"] for i in idx], dtype=float)
    labels = np.array(["ABCD".index(rows[i]["gold_letter"]) for i in idx])
    pred = probs.argmax(axis=1); wrong = (pred != labels).astype(int)
    p_gold = np.clip(probs[np.arange(len(idx)), labels], 1e-9, None)
    out = {"ACC": float((pred == labels).mean()), "NLL": float(-np.log(p_gold).mean())}
    out["ECE"], out["MCE"] = ece_mce(probs, labels)
    for s in SIGNALS:
        if s == "one_minus_maxprob":
            v = 1.0 - probs.max(axis=1)
        else:
            v = np.array([rows[i].get(s) if rows[i].get(s) is not None else np.nan for i in idx], dtype=float)
        out[f"auroc_{s}"] = auroc(wrong, v) if not np.all(np.isnan(v)) else float("nan")
    return out


def paired_bootstrap(rows_a, rows_b, n_boot, seed):
    """Join on id; -> {metric: {point, lo, hi}} for B - A with the SAME resampled
    examples in both arms."""
    ia = {r["id"]: r for r in rows_a}; ib = {r["id"]: r for r in rows_b}
    ids = [i for i in ia if i in ib]
    if len(ids) != len(ia) or len(ids) != len(ib):
        print(f"  WARNING: paired bootstrap on the {len(ids)} common ids (A has {len(ia)}, B has {len(ib)})")
    if not ids:
        return None
    A = [ia[i] for i in ids]; B = [ib[i] for i in ids]
    ma, mb = metrics_from_rows(A), metrics_from_rows(B)
    keys = [k for k in ma if not (np.isnan(ma[k]) or np.isnan(mb[k]))]
    point = {k: mb[k] - ma[k] for k in keys}
    rng = np.random.default_rng(seed); n = len(ids)
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ra, rb = metrics_from_rows(A, idx), metrics_from_rows(B, idx)
        for k in keys:
            d = rb[k] - ra[k]
            if not np.isnan(d):
                boots[k].append(d)
    out = {}
    for k in keys:
        if boots[k]:
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
        else:
            lo = hi = float("nan")
        out[k] = {"point": point[k], "lo": float(lo), "hi": float(hi), "n": n}
    return out


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def order_key(tag):
    for i, pref in enumerate(ROW_ORDER):
        if tag.startswith(pref):
            return (i, tag)
    return (len(ROW_ORDER), tag)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def auroc_cell(d):
    return "–" if not d else f"{fmt(d['auroc'])} [{fmt(d['lo'], 2)},{fmt(d['hi'], 2)}]"


def delta_cell(d):
    if not d:
        return "–"
    star = "*" if (d["lo"] > 0 or d["hi"] < 0) else ""
    return f"{d['point']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]{star}"


def msd(vals, nd=3):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f} ± {np.std(vals, ddof=1):.{nd}f}"


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.in_dir, f"*_{args.split}.json")))
    rows, per = {}, {}
    for f in files:
        tag = os.path.basename(f)[: -len(f"_{args.split}.json")]
        if args.tags and tag not in args.tags:
            continue
        rows[tag] = json.load(open(f))
        pth = os.path.join(args.in_dir, f"{tag}_{args.split}_perexample.jsonl")
        if os.path.exists(pth):
            per[tag] = [json.loads(l) for l in open(pth) if l.strip()]
    if not rows:
        raise SystemExit(f"no {args.in_dir}/*_{args.split}.json files found")
    tags = sorted(rows, key=order_key)

    L = [f"# Letter read-out comparison — split `{args.split}`", "",
         "Metrics on the answer letter only (4-way distribution at the `Answer:` position, identical prompt "
         "in both arms). AUROC(wrong): label 1 = wrong answer, higher signal ⇒ wrong; 95% example-level "
         "bootstrap CI. Arm A = answer-only training (`target_mode=letter`), arm B = answer + explanation "
         "(`target_mode=answer_explanation`); both on the same MedMCQA rows (shared eligible-ID list).", ""]

    # 1. per-row table
    hdr = ["tag", "method", "n", *SCALARS] + [f"AUROC {s}" for s in SIGNALS]
    L += ["## Rows", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    table = []
    for t in tags:
        r = rows[t]; cfg = r.get("config", {})
        cells = [t, cfg.get("method", "?"), str(r["n"])] + [fmt(r[k]) for k in SCALARS]
        cells += [auroc_cell(r.get("auroc_wrong", {}).get(s)) for s in SIGNALS]
        L.append("| " + " | ".join(cells) + " |")
        table.append({"tag": t, "method": cfg.get("method"), "seed": cfg.get("seed"), "n": r["n"],
                      **{k: r[k] for k in SCALARS}, "auroc_wrong": r.get("auroc_wrong", {}),
                      "file": os.path.join(args.in_dir, f"{t}_{args.split}.json")})

    # 2. paired deltas, exact tag match
    pairs = []
    for ta in tags:
        if not ta.startswith(ARM_A + "_"):
            continue
        tb = ARM_B + ta[len(ARM_A):]
        if tb not in rows:
            print(f"  (no arm-B partner for {ta}: expected tag {tb})"); continue
        if ta not in per or tb not in per:
            print(f"  (per-example files missing for {ta} / {tb}; cannot compute paired CI)"); continue
        d = paired_bootstrap(per[ta], per[tb], args.n_boot, args.boot_seed)
        if d:
            pairs.append({"arm_a": ta, "arm_b": tb, "setting": ta[len(ARM_A) + 1:], "delta": d})
    dkeys = SCALARS + [f"auroc_{s}" for s in SIGNALS]
    L += ["", "## Arm B − arm A (paired, exact same method / β / S / seed)", "",
          "Paired example-level bootstrap (same resampled examples in both arms, joined on id); "
          "`*` = 95% CI excludes 0.", "",
          "| setting | n | " + " | ".join(f"Δ{k}" for k in dkeys) + " |", "|---|---|" + "---|" * len(dkeys)]
    for p in pairs:
        n = next(iter(p["delta"].values()))["n"]
        L.append(f"| {p['setting']} | {n} | " + " | ".join(delta_cell(p["delta"].get(k)) for k in dkeys) + " |")
    if not pairs:
        L.append("| (no exactly matched A/B pairs found) |" + " |" * (len(dkeys) + 1))

    # 3. multi-seed summary (read-out seeds: fcvr MC sampling; kvq/det are deterministic)
    groups = defaultdict(list)
    for t in tags:
        groups[SEED_RE.sub("", t)].append(t)
    multi = {g: ts for g, ts in groups.items() if len(ts) > 1}
    seed_summary = {}
    if multi:
        L += ["", "## Across read-out seeds (mean ± sd over tags differing only in -s<seed>)", "",
              "| tag (seed stripped) | seeds | " + " | ".join(SCALARS) + " | " + " | ".join(f"AUROC {s}" for s in SIGNALS) + " |",
              "|---|---|" + "---|" * (len(SCALARS) + len(SIGNALS))]
        for g, ts in sorted(multi.items(), key=lambda kv: order_key(kv[0])):
            cells = [msd([rows[t][k] for t in ts]) for k in SCALARS]
            cells += [msd([rows[t].get("auroc_wrong", {}).get(s, {}).get("auroc", np.nan) for t in ts]) for s in SIGNALS]
            L.append(f"| {g} | {len(ts)} | " + " | ".join(cells) + " |")
            seed_summary[g] = {"tags": ts}
        pg = defaultdict(list)
        for p in pairs:
            pg[SEED_RE.sub("", p["setting"])].append(p)
        pm = {g: ps for g, ps in pg.items() if len(ps) > 1}
        if pm:
            L += ["", "| Δ setting (seed stripped) | seeds | " + " | ".join(f"Δ{k}" for k in dkeys) + " |",
                  "|---|---|" + "---|" * len(dkeys)]
            for g, ps in pm.items():
                cells = [msd([p["delta"][k]["point"] for p in ps if k in p["delta"]]) for k in dkeys]
                L.append(f"| {g} | {len(ps)} | " + " | ".join(cells) + " |")

    # 4. per-subject ACC / ECE
    per_subject = {}
    subj_tags = [t for t in tags if t.startswith("arm") and t in per and per[t] and per[t][0].get("subject_name")]
    if subj_tags:
        data = {}
        for t in subj_tags:
            by = defaultdict(list)
            for e in per[t]:
                by[e["subject_name"]].append(e)
            data[t] = by
        ref = next(iter(data.values()))
        subjects = sorted({s for by in data.values() for s in by}, key=lambda s: -len(ref.get(s, [])))
        L += ["", "## Per-subject ACC / ECE (arms only)", "",
              "| subject | n | " + " | ".join(f"{t} ACC/ECE" for t in data) + " |", "|---|---|" + "---|" * len(data)]
        for s in subjects:
            cells = []
            for t, by in data.items():
                exs = by.get(s, [])
                if len(exs) < 5:
                    cells.append("–"); continue
                m = metrics_from_rows(exs)
                cells.append(f"{m['ACC']:.3f}/{m['ECE']:.3f}")
                per_subject.setdefault(t, {})[s] = {"n": len(exs), "ACC": m["ACC"], "ECE": m["ECE"]}
            L.append(f"| {s} | {len(ref.get(s, []))} | " + " | ".join(cells) + " |")

    L += ["", "Files: " + ", ".join(os.path.basename(r["file"]) for r in table)]
    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, f"letter_arms_{args.split}.md")
    js = os.path.join(args.out_dir, f"letter_arms_{args.split}.json")
    with open(md, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(js, "w") as f:
        json.dump({"split": args.split, "rows": table, "paired_deltas_B_minus_A": pairs,
                   "seed_groups": seed_summary, "per_subject": per_subject}, f, indent=2)
    print("\n".join(L))
    print(f"\nwrote {md}\n      {js}")


if __name__ == "__main__":
    main()

"""Baseline ladder table (supervisor point 4): does FCVR genuinely help, or is
ILV just a complicated signal?

Collects, per ARM, the labelled sequence-level readouts and the input-level OoD
results and prints one comparison table:
    arms  = det (Stage-1, stock routers) | untrained FCVR heads | pretrained-prior FCVR
            | MAP-prior FCVR | KL mask none/attention/answer | beta=0   (+ any tag you pass)
    cols  = n, primary acc (whatever --label selects), probe acc, expl_entails rate, truncated rate,
            ROUGE-L / BERTScore / unigram-F1, tokens/s, wall-clock/example,
            AUROC(wrong) with bootstrap CI for the primary ILV score, entropy_max, gen_len,
            OoD AUROC (ilv_last / entropy_last) per OoD set.

Inputs are matched by filename tag:
    results/token_analysis/step1_<source>[_val]_generate_<TAG>_seqlevel_labeled.jsonl   (labels)
    results/token_analysis/step1_<source>[_val]_generate_<TAG>.json                     (config/runtime)
    results/input_level_ood/input_ood_<source>[_val]_<run_suffix>_<TAG>.json            (OoD, optional)
  (--source medexqa | medmcqa_gen; default medexqa)

Usage (CPU):
    python baseline_ladder_report.py --split val --tags beta0.01-S35-s42 det-S35-s42 untrained-S35-s42 ...
    python baseline_ladder_report.py --split test --tags ... --primary_score ilv_online_mean_last10
Writes results/reports/ladder_<split>.md and .json (never overwrites the inputs).
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

from uq_stats import auroc, bootstrap_ci


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def label_source_counts(rows):
    """Provenance tally. Labelled files written before `label_source` existed carry no
    such key; those rows are counted under an explicit string rather than None, so the
    counts stay sortable (str vs None does not compare) and JSON-safe."""
    return dict(Counter(r.get("label_source") or "(not recorded)" for r in rows))


def summarize_arm(tag, split, args):
    split_tok = "" if split == "test" else f"_{split}"
    lab = find_one(f"{args.token_dir}/step1_{args.source}{split_tok}_generate_{tag}_seqlevel_labeled.jsonl")
    if lab is None:
        return {"tag": tag, "missing": f"{args.token_dir}/step1_{args.source}{split_tok}_generate_{tag}_seqlevel_labeled.jsonl"}
    cfg_path = find_one(f"{args.token_dir}/step1_{args.source}{split_tok}_generate_{tag}.json")
    cfg = json.load(open(cfg_path)) if cfg_path else {}
    rows = load_rows(lab)
    label = args.label
    lab_rows = [r for r in rows if r.get(label) is not None]
    y = np.array([0 if r[label] else 1 for r in lab_rows], dtype=int)   # 1 = wrong

    def mean_of(key, rs=rows):
        v = [r[key] for r in rs if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    def score_auc(name):
        s = [r["scores"].get(name) for r in lab_rows]
        if any(v is None for v in s) or not s:
            return None
        return bootstrap_ci(auroc, y, np.array(s, dtype=float), n_boot=args.n_boot)

    out = {
        "tag": tag, "arm": cfg.get("config", {}).get("arm"), "routing": cfg.get("config", {}).get("routing"),
        "run_suffix": cfg.get("config", {}).get("run_suffix"), "prior_source": cfg.get("config", {}).get("prior_source"),
        "n": len(rows), "n_labeled": len(lab_rows), "n_unlabeled": len(rows) - len(lab_rows), "label": label,
        "label_source_counts": label_source_counts(rows),
        # Accuracy of whatever --label selects, over the SAME rows the AUROC uses,
        # so it shares a denominator with n_labeled.
        "primary_acc": mean_of(label, lab_rows), "probe_acc": mean_of("correct_probe"),
        "expl_entails_rate": mean_of("expl_entails"), "expl_any_contra_rate": mean_of("expl_any_contra"),
        "truncated_rate": mean_of("truncated"), "unjudgeable_rate": mean_of("unjudgeable"),
        "rouge_l": mean_of("rouge_l"), "bertscore_f1": mean_of("bertscore_f1"), "expl_f1": mean_of("expl_f1"),
        "bleu": mean_of("bleu"), "meteor": mean_of("meteor"),
        "tokens_per_second": mean_of("tokens_per_second"), "gen_seconds": mean_of("gen_seconds"),
        "auroc": {
            args.primary_score: score_auc(args.primary_score),
            "ilv_posthoc_mean_last10": score_auc("ilv_posthoc_mean_last10"),
            "entropy_max_BASELINE": score_auc("entropy_max_BASELINE"),
            "nll_per_token_BASELINE": score_auc("nll_per_token_BASELINE"),
            "gen_len_BASELINE": score_auc("gen_len_BASELINE"),
            "gate_entropy_mean_BASELINE": score_auc("gate_entropy_mean_BASELINE"),
        },
        "ood": {},
    }
    # Per-subject breakdown (medmcqa_gen rows carry subject_name; MedExQA rows do not):
    # primary accuracy + primary-score AUROC per subject (JSON only; n per subject is small).
    if any(r.get("subject_name") for r in lab_rows):
        by_subj = {}
        for r, yy in zip(lab_rows, y):
            by_subj.setdefault(r.get("subject_name") or "?", []).append((r, int(yy)))
        out["per_subject"] = {}
        for subj, items in sorted(by_subj.items(), key=lambda kv: -len(kv[1])):
            ys = np.array([yy for _, yy in items], dtype=int)
            ss = [r["scores"].get(args.primary_score) for r, _ in items]
            auc = None
            if len(items) >= 10 and 0 < ys.sum() < len(ys) and all(v is not None for v in ss):
                auc = float(auroc(ys, np.array(ss, dtype=float)))
            out["per_subject"][subj] = {"n": len(items), "primary_acc": float(1.0 - ys.mean()),
                                        f"auroc_{args.primary_score}": auc}
    ood_path = find_one(f"{args.ood_dir}/input_ood_{args.source}{split_tok}_*_{tag}.json")
    if ood_path:
        o = json.load(open(ood_path))
        for code, r in o.get("ood", {}).items():
            out["ood"][code] = {sig: r[sig]["auroc"] for sig in r if "auroc" in r[sig]}
    return out


def fmt_ci(d):
    if not d or d.get("point") is None or np.isnan(d["point"]):
        return "—"
    if np.isnan(d.get("lo", np.nan)):
        return f"{d['point']:.3f}"
    return f"{d['point']:.3f} [{d['lo']:.2f},{d['hi']:.2f}]"


def fmt(v, nd=3):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--source", default="medexqa", choices=["medexqa", "medmcqa_gen"],
                   help="Generation dataset the readouts were run on (filename prefix step1_<source>/input_ood_<source>).")
    p.add_argument("--tags", nargs="+", required=True, help="Run tags in the order the rows should appear.")
    p.add_argument("--label", default="correct_primary",
                   help="Correctness label column (pre-registered: correct_primary = option_correct w/ probe fallback).")
    p.add_argument("--primary_score", default="ilv_online_mean_last10")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--token_dir", default="results/token_analysis")
    p.add_argument("--ood_dir", default="results/input_level_ood")
    p.add_argument("--out_dir", default="results/reports")
    p.add_argument("--out_name", default=None)
    args = p.parse_args()

    arms = [summarize_arm(t, args.split, args) for t in args.tags]
    os.makedirs(args.out_dir, exist_ok=True)
    # MedExQA keeps the historical name ladder_<split>; other sources are prefixed so reports never collide.
    name = args.out_name or (f"ladder_{args.split}" if args.source == "medexqa" else f"ladder_{args.source}_{args.split}")
    with open(os.path.join(args.out_dir, name + ".json"), "w") as f:
        json.dump({"split": args.split, "label": args.label, "primary_score": args.primary_score, "arms": arms}, f, indent=2)

    ood_codes = sorted({c for a in arms for c in a.get("ood", {})})
    hdr = ["tag", "arm", "n", "n lab", "n unlab", "primary acc", "probe acc", "entails", "trunc", "ROUGE-L", "BERTSc", "uniF1",
           "tok/s", f"AUROC {args.primary_score}", "AUROC ent_max", "AUROC nll", "AUROC len"] + [f"OoD {c} ilv/ent" for c in ood_codes]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for a in arms:
        if "missing" in a:
            lines.append(f"| {a['tag']} | MISSING: {a['missing']} |" + " |" * (len(hdr) - 2))
            continue
        row = [a["tag"], fmt(a["arm"]), str(a["n"]), str(a["n_labeled"]), str(a["n_unlabeled"]),
               fmt(a["primary_acc"]), fmt(a["probe_acc"]), fmt(a["expl_entails_rate"]),
               fmt(a["truncated_rate"], 2), fmt(a["rouge_l"]), fmt(a["bertscore_f1"]), fmt(a["expl_f1"]),
               fmt(a["tokens_per_second"], 1), fmt_ci(a["auroc"].get(args.primary_score)),
               fmt_ci(a["auroc"].get("entropy_max_BASELINE")), fmt_ci(a["auroc"].get("nll_per_token_BASELINE")),
               fmt_ci(a["auroc"].get("gen_len_BASELINE"))]
        for c in ood_codes:
            o = a["ood"].get(c, {})
            row.append(f"{fmt(o.get('ilv_last'))}/{fmt(o.get('entropy_last'))}")
        lines.append("| " + " | ".join(row) + " |")
    # Provenance of the primary label, so a probe-fallback-contaminated table can never
    # be mistaken for one graded purely on what the model actually generated.
    prov = ["", "## Primary-label provenance", "",
            "`n unlab` rows carry no primary label and are excluded from every AUROC above.", ""]
    for a in arms:
        if "missing" in a:
            continue
        counts = ", ".join(f"{k}={v}" for k, v in sorted(a["label_source_counts"].items()))
        prov.append(f"- **{a['tag']}**: {counts or '(none recorded)'}")
    md = (f"# Baseline ladder — source={args.source}, split={args.split}, label={args.label}, primary score={args.primary_score}\n\n"
          "AUROC = predict WRONG (sign fixed a priori: higher score ⇒ wrong); [lo,hi] = 95% example-level bootstrap CI. "
          "OoD columns = input-level AUROC(OoD=1) for ilv_last / entropy_last.\n\n"
          + "\n".join(lines) + "\n" + "\n".join(prov) + "\n")
    with open(os.path.join(args.out_dir, name + ".md"), "w") as f:
        f.write(md)
    print(md)
    print(f"Saved: {os.path.join(args.out_dir, name)}.md/.json")


if __name__ == "__main__":
    main()

"""Comparison table for the MedMCQA answer-only vs answer+explanation study
(branch MedMCQA-comparison). Collects every evaluate_letter.py aggregate for one
split and prints / writes one table:

    rows = one per evaluate_letter.py --tag (zero-shot, OBQA reference, arm A /
           arm B x {kvq_ft, det, fcvr}), in a fixed order
    cols = n, ACC, NLL, ECE, MCE, AUROC(wrong) [CI] per signal
           (letter_entropy, one_minus_maxprob, gate_entropy_last, ilv_last)
plus an arm-A vs arm-B delta block per method and a per-subject ACC/ECE table
(from the per-example files; medmcqa_gen only).

Usage (CPU):
    python letter_eval_report.py --split test
    python letter_eval_report.py --split val --in_dir results/letter_eval --out_dir results/reports
Writes <out_dir>/letter_arms_<split>.md and .json; never touches the inputs.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

SIGNALS = ["letter_entropy", "one_minus_maxprob", "gate_entropy_last", "gate_entropy_last_fcvr", "ilv_last"]
# Preferred row order (prefix match on the tag); anything else follows alphabetically.
ROW_ORDER = ["zero-shot", "ref-obqa", "armA-letter_kvq", "armA-letter_det", "armA-letter_fcvr",
             "armB-ansexp_kvq", "armB-ansexp_det", "armB-ansexp_fcvr"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--in_dir", type=str, default="results/letter_eval")
    p.add_argument("--out_dir", type=str, default="results/reports")
    p.add_argument("--tags", type=str, nargs="*", default=None, help="Restrict to these tags (default: all found).")
    return p.parse_args()


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
    if not d:
        return "–"
    return f"{fmt(d['auroc'])} [{fmt(d['lo'], 2)},{fmt(d['hi'], 2)}]"


def ece_of(probs, labels, n_bins=10):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); acc = (pred == labels)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return float(ece)


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.in_dir, f"*_{args.split}.json")))
    rows = {}
    for f in files:
        tag = os.path.basename(f)[: -len(f"_{args.split}.json")]
        if args.tags and tag not in args.tags:
            continue
        with open(f) as fh:
            rows[tag] = json.load(fh)
    if not rows:
        raise SystemExit(f"no {args.in_dir}/*_{args.split}.json files found")
    tags = sorted(rows, key=order_key)

    lines = [f"# Letter read-out comparison — split `{args.split}`", "",
             "Metrics on the answer letter only (4-way distribution at the `Answer:` position). "
             "AUROC(wrong): label 1 = wrong answer, higher signal ⇒ wrong; 95% bootstrap CI. "
             "Arm A = answer-only training (`target_mode=letter`), arm B = answer + explanation "
             "(`target_mode=answer_explanation`); both on the same MedMCQA rows.", ""]
    hdr = ["tag", "method", "prompt arm", "n", "ACC", "NLL", "ECE", "MCE"] + [f"AUROC {s}" for s in SIGNALS]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    table = []
    for t in tags:
        r = rows[t]; cfg = r.get("config", {})
        cells = [t, cfg.get("method", "?"), cfg.get("target_mode", "?"), str(r["n"]),
                 fmt(r["ACC"]), fmt(r["NLL"]), fmt(r["ECE"]), fmt(r["MCE"])]
        cells += [auroc_cell(r.get("auroc_wrong", {}).get(s)) for s in SIGNALS]
        lines.append("| " + " | ".join(cells) + " |")
        table.append({"tag": t, "method": cfg.get("method"), "target_mode": cfg.get("target_mode"),
                      "n": r["n"], "ACC": r["ACC"], "NLL": r["NLL"], "ECE": r["ECE"], "MCE": r["MCE"],
                      "auroc_wrong": r.get("auroc_wrong", {}), "file": os.path.join(args.in_dir, f"{t}_{args.split}.json")})

    # arm A vs arm B deltas per method (B - A)
    deltas = []
    lines += ["", "## Arm B − arm A (same method)", "",
              "| method | ΔACC | ΔNLL | ΔECE | ΔMCE | " + " | ".join(f"ΔAUROC {s}" for s in SIGNALS) + " |",
              "|---|---|---|---|---|" + "---|" * len(SIGNALS)]
    for meth in ["kvq", "det", "fcvr"]:
        a = next((rows[t] for t in tags if t.startswith(f"armA-letter_{meth}")), None)
        b = next((rows[t] for t in tags if t.startswith(f"armB-ansexp_{meth}")), None)
        if not a or not b:
            continue
        d = {k: b[k] - a[k] for k in ("ACC", "NLL", "ECE", "MCE")}
        da = {}
        for s in SIGNALS:
            ea, eb = a.get("auroc_wrong", {}).get(s), b.get("auroc_wrong", {}).get(s)
            da[s] = (eb["auroc"] - ea["auroc"]) if (ea and eb) else None
        lines.append(f"| {meth} | {d['ACC']:+.3f} | {d['NLL']:+.3f} | {d['ECE']:+.3f} | {d['MCE']:+.3f} | "
                     + " | ".join(("–" if da[s] is None else f"{da[s]:+.3f}") for s in SIGNALS) + " |")
        deltas.append({"method": meth, **d, "auroc_wrong": da})

    # per-subject ACC / ECE from the per-example files (medmcqa_gen)
    per_subject = {}
    subj_tags = [t for t in tags if t.startswith("arm")]
    if subj_tags:
        lines += ["", "## Per-subject ACC / ECE (arms only)", ""]
        data = {}
        for t in subj_tags:
            pth = os.path.join(args.in_dir, f"{t}_{args.split}_perexample.jsonl")
            if not os.path.exists(pth):
                continue
            ex = [json.loads(l) for l in open(pth) if l.strip()]
            if not ex or ex[0].get("subject_name") is None:
                continue
            by = defaultdict(list)
            for e in ex:
                by[e["subject_name"]].append(e)
            data[t] = by
        if data:
            subjects = sorted({s for by in data.values() for s in by}, key=lambda s: -len(next(iter(data.values())).get(s, [])))
            lines.append("| subject | n | " + " | ".join(f"{t} ACC/ECE" for t in data) + " |")
            lines.append("|---|---|" + "---|" * len(data))
            for s in subjects:
                n = len(next(iter(data.values())).get(s, []))
                cells = []
                for t, by in data.items():
                    exs = by.get(s, [])
                    if len(exs) < 5:
                        cells.append("–"); continue
                    probs = np.array([e["probs"] for e in exs]); labels = np.array(["ABCD".index(e["gold_letter"]) for e in exs])
                    acc = float((probs.argmax(1) == labels).mean()); ece = ece_of(probs, labels)
                    cells.append(f"{acc:.3f}/{ece:.3f}")
                    per_subject.setdefault(t, {})[s] = {"n": len(exs), "ACC": acc, "ECE": ece}
                lines.append(f"| {s} | {n} | " + " | ".join(cells) + " |")

    lines += ["", "Files: " + ", ".join(os.path.basename(r["file"]) for r in table)]
    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, f"letter_arms_{args.split}.md")
    js = os.path.join(args.out_dir, f"letter_arms_{args.split}.json")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(js, "w") as f:
        json.dump({"split": args.split, "rows": table, "deltas_B_minus_A": deltas, "per_subject": per_subject}, f, indent=2)
    print("\n".join(lines))
    print(f"\nwrote {md}\n      {js}")


if __name__ == "__main__":
    main()

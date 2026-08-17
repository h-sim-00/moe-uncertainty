"""Correctness / quality labels for generated MedExQA explanations (supervisor point 1).

Reads a `*_seqlevel.jsonl` written by `analyze_token_signals.py --generate` and
adds INDEPENDENT labels per generation. Nothing here folds several judgements
into one verdict -- each question below gets its own field:

  Was the selected answer option correct?      option_correct  (1 / 0 / None)
      `committed_letter` = the option the generated text itself commits to
      (explicit "the answer is B" style phrases, else a match against the
      option texts). None -> option_unjudgeable.
  Did the explanation agree with the refs?      expl_entail_frac, expl_entails
  Did it contradict the refs?                   expl_contra_frac, expl_any_contra
      Sentence-level NLI (cross-encoder) of EACH generated sentence against
      EACH gold explanation. A single entailed sentence never rescues an
      explanation that also contains a contradicted one. SECONDARY signal;
      calibrate it with --audit_export / --audit_import (25-50 hand labels).
  Was it impossible to judge?                   unjudgeable
  Was it cut off at max_new_tokens?             truncated (copied from the readout)
  Letter-probe (separate MCQA pass):            correct_probe (copied; SECONDARY)

Explanation quality (multi-reference max): unigram-F1 (kept), BLEU (sacrebleu),
ROUGE-L, METEOR, BERTScore-F1 with SciBERT -- MedExQA's own metric suite.

Pre-registered PRIMARY abstention label = `option_correct`; when the option is
unjudgeable the row falls back to `correct_probe` and says so in
`label_source`. Both are written; nothing is silently merged.

Usage (quail, moe_env):
    python label_generation_correctness.py --input results/token_analysis/<...>_seqlevel.jsonl
        [--nli_model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli] [--no_nli]
        [--no_metrics] [--audit_export 40] [--audit_import results/labels/audit_<tag>.csv]
Writes <input stem>_labeled.jsonl + <stem>_labels_summary.json (never overwrites the input).
Deps for the metric suite: pip install sacrebleu rouge-score nltk bert-score
"""
import argparse
import csv
import json
import os
import re
import string
import sys
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Option commitment: which option does the generated explanation itself pick?
# ---------------------------------------------------------------------------
_LETTER = r"\(?([A-D])\)?"
_EXPLICIT_PATTERNS = [
    # "the correct answer is B", "answer: (C)", "correct option would be D", "the right choice is A"
    re.compile(r"\b(?:correct|right|best)?\s*(?:answer|option|choice)\s*(?:is|was|would be|should be|:|-)\s*" + _LETTER + r"(?![A-Za-z])", re.I),
    # "B is the correct answer", "(C) is correct"
    re.compile(r"(?<![A-Za-z])" + _LETTER + r"\s+is\s+(?:the\s+)?(?:correct|right|best)(?:\s+(?:answer|option|choice))?\b", re.I),
    # "option B", "choice C" (only when followed by a comma/period/is/because -> a statement, not "option B is wrong")
    re.compile(r"\b(?:option|choice)\s+" + _LETTER + r"(?=\s*(?:[,.:;]|is\s+(?:the\s+)?correct|because|since|as\b))", re.I),
    # leading "B." / "B)" / "(B)" at the very start
    re.compile(r"^\s*\(?([A-D])[\.\):]\s", re.I),
]
_NEGATED = re.compile(r"\b(?:not|isn't|is not|incorrect|wrong)\b", re.I)


def parse_options(question_text):
    """Parse 'A. text' option lines from either MedExQA prompt format."""
    if not question_text:
        return {}
    opts = {}
    for m in re.finditer(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$", question_text):
        opts[m.group(1).upper()] = m.group(2).strip()
    return opts


def _norm(s):
    return " ".join(w.strip(string.punctuation).lower() for w in s.split() if w.strip(string.punctuation))


def committed_option(gen_text, options):
    """Return (letter or None, how). Explicit letter phrases first (window = the
    first 400 chars, where the commitment is stated), then option-text matching
    in the first two sentences (longest unique match wins)."""
    if not gen_text:
        return None, None
    head = gen_text[:400]
    for pat in _EXPLICIT_PATTERNS:
        for m in pat.finditer(head):
            # skip "the answer is not B" style negations right before/inside the match
            pre = head[max(0, m.start() - 12):m.start()]
            if _NEGATED.search(pre) or _NEGATED.search(m.group(0)):
                continue
            return m.group(1).upper(), "explicit_letter"
    if options:
        first = " ".join(re.split(r"(?<=[.!?])\s+", gen_text.strip())[:2])
        nf = _norm(first)
        hits = []
        for letter, txt in options.items():
            nt = _norm(txt)
            if not nt:
                continue
            if len(nt.split()) >= 2 and nt in nf:
                hits.append((len(nt), letter))
            elif len(nt.split()) == 1 and re.search(r"\b" + re.escape(nt) + r"\b", nf):
                hits.append((len(nt), letter))
        # Exactly ONE option's text mentioned up front -> that is the commitment.
        # Several options named (comparisons, lists) -> ambiguous -> None.
        if len(hits) == 1:
            return hits[0][1], "option_text"
    return None, None


# ---------------------------------------------------------------------------
# Sentence-level NLI
# ---------------------------------------------------------------------------
def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


class NLIScorer:
    def __init__(self, model_name, device="cuda:0", batch_size=32, max_len=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device, self.bs, self.max_len = device, batch_size, max_len
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for i, n in id2label.items() if n.startswith("entail"))
        self.i_con = next(i for i, n in id2label.items() if n.startswith("contra"))
        self.i_neu = next((i for i, n in id2label.items() if n.startswith("neutral")), None)
        print(f"NLI model {model_name}: labels={id2label}")

    def probs(self, premises, hypotheses):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(premises), self.bs):
                enc = self.tok(premises[i:i + self.bs], hypotheses[i:i + self.bs], return_tensors="pt",
                               padding=True, truncation="only_first", max_length=self.max_len).to(self.device)
                p = self.torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
                out.append(p)
        return np.concatenate(out, axis=0) if out else np.zeros((0, 3))


def nli_labels(scorer, gen_text, refs, entail_thr, contra_thr, frac_thr):
    """Per-sentence NLI vs each reference -> independent explanation fields."""
    sents = split_sentences(gen_text)
    res = {"n_sentences": len(sents), "expl_entail_frac": None, "expl_contra_frac": None,
           "expl_any_contra": None, "expl_entails": None, "expl_neutral": None,
           "nli_max_entail": None, "nli_max_contra": None, "nli_sentence_verdicts": None}
    if not sents or not refs:
        return res
    prem, hyp = [], []
    for s_ in sents:
        for r in refs:
            prem.append(r); hyp.append(s_)
    P = scorer.probs(prem, hyp).reshape(len(sents), len(refs), -1)
    ent = P[:, :, scorer.i_ent].max(axis=1)   # best supporting ref per sentence
    con = P[:, :, scorer.i_con].max(axis=1)   # worst contradicting ref per sentence
    verdicts = []
    for e, c in zip(ent, con):
        if c >= contra_thr and c > e:
            verdicts.append("contra")
        elif e >= entail_thr:
            verdicts.append("entail")
        else:
            verdicts.append("neutral")
    ef = float(np.mean([v == "entail" for v in verdicts]))
    cf = float(np.mean([v == "contra" for v in verdicts]))
    res.update({
        "expl_entail_frac": ef, "expl_contra_frac": cf,
        "expl_any_contra": bool(cf > 0),
        # entails ONLY if enough sentences are supported AND none is contradicted
        "expl_entails": bool(ef >= frac_thr and cf == 0),
        "expl_neutral": bool(all(v == "neutral" for v in verdicts)),
        "nli_max_entail": float(ent.max()), "nli_max_contra": float(con.max()),
        "nli_sentence_verdicts": verdicts,
    })
    return res


# ---------------------------------------------------------------------------
# Explanation-quality metrics (MedExQA suite), multi-reference max
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self, want, bertscore_model, device):
        self.avail = {}
        if not want:
            return
        try:
            import sacrebleu; self.sacrebleu = sacrebleu; self.avail["bleu"] = True
        except Exception as e:
            print(f"(BLEU unavailable: {e!r} -- pip install sacrebleu)")
        try:
            from rouge_score import rouge_scorer
            self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True); self.avail["rouge_l"] = True
        except Exception as e:
            print(f"(ROUGE-L unavailable: {e!r} -- pip install rouge-score)")
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score
            for res in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
                try:
                    nltk.download(res, quiet=True)
                except Exception:
                    pass
            self.meteor_score = meteor_score; self.nltk = nltk; self.avail["meteor"] = True
        except Exception as e:
            print(f"(METEOR unavailable: {e!r} -- pip install nltk)")
        try:
            import bert_score
            self.bert_score = bert_score; self.bertscore_model = bertscore_model; self.device = device
            self.avail["bertscore_f1"] = True
        except Exception as e:
            print(f"(BERTScore unavailable: {e!r} -- pip install bert-score)")

    def per_row(self, gen, refs):
        out = {}
        if self.avail.get("bleu"):
            out["bleu"] = max(self.sacrebleu.sentence_bleu(gen, [r]).score for r in refs) / 100.0
        if self.avail.get("rouge_l"):
            out["rouge_l"] = max(self.rouge.score(r, gen)["rougeL"].fmeasure for r in refs)
        if self.avail.get("meteor"):
            try:
                gt = self.nltk.word_tokenize(gen)
                out["meteor"] = max(self.meteor_score([self.nltk.word_tokenize(r)], gt) for r in refs)
            except Exception:
                out["meteor"] = max(self.meteor_score([r.split()], gen.split()) for r in refs)
        return out

    def bertscore_batch(self, gens, refs_list):
        """Multi-ref max via bert_score's list-of-lists reference support."""
        if not self.avail.get("bertscore_f1"):
            return [None] * len(gens)
        _, _, F = self.bert_score.score(gens, refs_list, model_type=self.bertscore_model,
                                        device=self.device, verbose=False, batch_size=32)
        return [float(f) for f in F.tolist()]


def _unigram_f1(pred, ref):
    tok = lambda s: [w.strip(string.punctuation).lower() for w in s.split()]
    p = [w for w in tok(pred) if w]; r = [w for w in tok(ref) if w]
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="*_seqlevel.jsonl from analyze_token_signals.py --generate")
    p.add_argument("--nli_model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    p.add_argument("--no_nli", action="store_true")
    p.add_argument("--nli_entail_thr", type=float, default=0.5)
    p.add_argument("--nli_contra_thr", type=float, default=0.5)
    p.add_argument("--nli_frac_thr", type=float, default=0.5,
                   help="expl_entails requires >= this fraction of sentences entailed AND zero contradicted.")
    p.add_argument("--no_metrics", action="store_true", help="Skip BLEU/ROUGE-L/METEOR/BERTScore.")
    p.add_argument("--bertscore_model", default="allenai/scibert_scivocab_uncased")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--audit_export", type=int, default=0,
                   help="Write the first N rows to results/labels/audit_<tag>.csv for hand-labelling (val only).")
    p.add_argument("--audit_import", type=str, default=None,
                   help="Filled-in audit CSV -> precision/recall of option/NLI labels vs human labels.")
    p.add_argument("--audit_dir", default="results/labels")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing _labeled.jsonl.")
    return p.parse_args()


def _agree(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"n": 0}
    return {"n": len(pairs), "agreement": float(np.mean([x == y for x, y in pairs])),
            "confusion": {f"{x}->{y}": int(sum(1 for xx, yy in pairs if xx == x and yy == y))
                          for x in (True, False) for y in (True, False)}}


def main():
    args = parse_args()
    if not args.input.endswith("_seqlevel.jsonl"):
        print("WARNING: expected a *_seqlevel.jsonl input", file=sys.stderr)
    stem = args.input[:-len(".jsonl")] if args.input.endswith(".jsonl") else args.input
    out_jsonl, out_summary = stem + "_labeled.jsonl", stem + "_labels_summary.json"
    if os.path.exists(out_jsonl) and not args.overwrite:
        sys.exit(f"Refusing to overwrite {out_jsonl} (pass --overwrite).")
    tag = os.path.basename(stem).replace("step1_medexqa_", "").replace("_seqlevel", "")

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"Loaded {len(rows)} generations from {args.input}")
    if rows and "gen_text" not in rows[0]:
        sys.exit("Input rows lack gen_text/refs -- regenerate with the iter1 analyze_token_signals.py.")

    scorer = None
    if not args.no_nli:
        try:
            scorer = NLIScorer(args.nli_model, device=args.device)
        except Exception as e:
            print(f"NLI model unavailable ({e!r}); NLI fields will be None.")
    metrics = Metrics(not args.no_metrics, args.bertscore_model, args.device)

    for r in rows:
        gen, refs = r.get("gen_text") or "", r.get("refs") or []
        gold = (r.get("gold_letter") or "").upper() or None
        opts = parse_options(r.get("question"))
        letter, how = committed_option(gen, opts)
        r["committed_letter"], r["committed_via"] = letter, how
        r["option_unjudgeable"] = letter is None
        r["option_correct"] = None if (letter is None or gold is None) else int(letter == gold)
        if scorer is not None:
            r.update(nli_labels(scorer, gen, refs, args.nli_entail_thr, args.nli_contra_thr, args.nli_frac_thr))
        else:
            r.update({"expl_entail_frac": None, "expl_contra_frac": None, "expl_any_contra": None,
                      "expl_entails": None, "expl_neutral": None, "n_sentences": len(split_sentences(gen))})
        r["unjudgeable"] = bool(r["option_unjudgeable"] and (r.get("expl_neutral") in (True, None)))
        r["truncated"] = bool(r.get("truncated"))
        # Pre-registered primary label with explicit provenance.
        if r["option_correct"] is not None:
            r["correct_primary"], r["label_source"] = bool(r["option_correct"]), "option_correct"
        elif r.get("correct_probe") is not None:
            r["correct_primary"], r["label_source"] = bool(r["correct_probe"]), "correct_probe(fallback)"
        else:
            r["correct_primary"], r["label_source"] = None, "none"
        r["wrong_primary"] = None if r["correct_primary"] is None else int(not r["correct_primary"])
        r["expl_f1"] = max((_unigram_f1(gen, x) for x in refs), default=0.0)
        r.update(metrics.per_row(gen, refs) if refs else {})
    if not args.no_metrics and rows:
        bs = metrics.bertscore_batch([r.get("gen_text") or "" for r in rows], [r.get("refs") or [""] for r in rows])
        for r, f in zip(rows, bs):
            if f is not None:
                r["bertscore_f1"] = f

    # ---- summary ----
    def _mean(key, pred=lambda r: True):
        v = [r[key] for r in rows if pred(r) and r.get(key) is not None]
        return float(np.mean(v)) if v else None
    summary = {
        "input": args.input, "n": len(rows), "tag": tag,
        "nli_model": None if scorer is None else args.nli_model,
        "thresholds": {"entail": args.nli_entail_thr, "contra": args.nli_contra_thr, "frac": args.nli_frac_thr},
        "counts": {
            "option_judgeable": int(sum(1 for r in rows if not r["option_unjudgeable"])),
            "committed_via": dict(Counter(r["committed_via"] for r in rows)),
            "unjudgeable": int(sum(1 for r in rows if r["unjudgeable"])),
            "truncated": int(sum(1 for r in rows if r["truncated"])),
            "label_source": dict(Counter(r["label_source"] for r in rows)),
        },
        "accuracy": {
            "option_correct (judgeable only)": _mean("option_correct"),
            "correct_probe": _mean("correct_probe"),
            "correct_primary": _mean("correct_primary"),
            "expl_entails": _mean("expl_entails"),
            "expl_any_contra": _mean("expl_any_contra"),
        },
        "agreement": {
            "option_vs_probe": _agree([None if r["option_correct"] is None else bool(r["option_correct"]) for r in rows],
                                      [r.get("correct_probe") for r in rows]),
            "option_vs_expl_entails": _agree([None if r["option_correct"] is None else bool(r["option_correct"]) for r in rows],
                                             [r.get("expl_entails") for r in rows]),
            "probe_vs_expl_entails": _agree([r.get("correct_probe") for r in rows], [r.get("expl_entails") for r in rows]),
        },
        "quality_means": {k: _mean(k) for k in ("expl_f1", "bleu", "rouge_l", "meteor", "bertscore_f1")},
        "quality_means_by_option_correct": {
            str(v): {k: _mean(k, lambda r, v=v: r["option_correct"] == v) for k in ("expl_f1", "rouge_l", "bertscore_f1")}
            for v in (1, 0)},
    }

    # ---- audit export / import ----
    if args.audit_export:
        os.makedirs(args.audit_dir, exist_ok=True)
        path = os.path.join(args.audit_dir, f"audit_{tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "question", "gold_letter", "ref_1", "ref_2", "gen_text",
                        "committed_letter", "option_correct", "expl_entail_frac", "expl_any_contra", "expl_entails",
                        "truncated", "human_option_letter", "human_option_correct", "human_expl_ok",
                        "human_expl_contradicts", "notes"])
            for r in rows[:args.audit_export]:
                refs = r.get("refs") or []
                w.writerow([r.get("id"), r.get("question"), r.get("gold_letter"), refs[0] if refs else "",
                            refs[1] if len(refs) > 1 else "", r.get("gen_text"), r["committed_letter"],
                            r["option_correct"], r.get("expl_entail_frac"), r.get("expl_any_contra"),
                            r.get("expl_entails"), r["truncated"], "", "", "", "", ""])
        summary["audit_export"] = path
        print(f"Audit sheet -> {path}  (fill human_* columns: option letter A-D, 1/0 flags; then --audit_import)")
    if args.audit_import:
        by_id = {r.get("id"): r for r in rows}
        h_opt, m_opt, h_ok, m_ent, h_con, m_con, h_letter, m_letter = [], [], [], [], [], [], [], []
        with open(args.audit_import, newline="", encoding="utf-8") as f:
            for a in csv.DictReader(f):
                r = by_id.get(a["id"])
                if r is None:
                    continue
                if a.get("human_option_correct", "").strip() in ("0", "1"):
                    h_opt.append(a["human_option_correct"].strip() == "1")
                    m_opt.append(None if r["option_correct"] is None else bool(r["option_correct"]))
                if a.get("human_option_letter", "").strip():
                    h_letter.append(a["human_option_letter"].strip().upper()); m_letter.append(r["committed_letter"])
                if a.get("human_expl_ok", "").strip() in ("0", "1"):
                    h_ok.append(a["human_expl_ok"].strip() == "1"); m_ent.append(r.get("expl_entails"))
                if a.get("human_expl_contradicts", "").strip() in ("0", "1"):
                    h_con.append(a["human_expl_contradicts"].strip() == "1"); m_con.append(r.get("expl_any_contra"))

        def prf(h, m):
            pairs = [(x, y) for x, y in zip(h, m) if y is not None]
            if not pairs:
                return {"n": 0}
            tp = sum(1 for x, y in pairs if x and y); fp = sum(1 for x, y in pairs if (not x) and y)
            fn = sum(1 for x, y in pairs if x and (not y))
            return {"n": len(pairs), "coverage": len(pairs) / max(1, len(h)),
                    "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                    "accuracy": float(np.mean([x == y for x, y in pairs]))}
        summary["audit"] = {
            "n_rows": len(h_opt), "option_correct_vs_human": prf(h_opt, m_opt),
            "committed_letter_vs_human": {"n": len(h_letter),
                                          "accuracy": float(np.mean([x == y for x, y in zip(h_letter, m_letter)])) if h_letter else None,
                                          "coverage": float(np.mean([y is not None for y in m_letter])) if m_letter else None},
            "expl_entails_vs_human_ok": prf(h_ok, m_ent),
            "expl_any_contra_vs_human_contradicts": prf(h_con, m_con),
        }
        print("Audit vs human:", json.dumps(summary["audit"], indent=2))

    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("counts", "accuracy", "agreement", "quality_means")}, indent=2))
    print(f"Saved: {out_jsonl}\n       {out_summary}")


if __name__ == "__main__":
    main()

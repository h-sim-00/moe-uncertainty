"""
Step 1 -- Does the FCVR (VGLR-FC) Inf-Logit-Var router signal transfer to
per-token generation?

Motivation
----------
The whole "UQ-guided decoding" plan rests on ONE untested assumption: that the
FCVR router's Inf-Logit-Var signal -- tr(posterior cov) = ||L||_F^2, computed
per token from the router's Cholesky factor -- actually spikes on the SAME
tokens a decoder cares about (numbers, named entities, logical connectives; more
generally, high predictive-entropy / high-surprisal tokens). The router heads
were trained ONLY on single-token MCQA argmax scoring, so whether the signal
means anything token-by-token over running text is an open question (critique
flaw #5: MCQA-trained heads -> long-form domain gap).

This script tests exactly that. It runs the trained Granite+FCVR model over text
(free-form prose by default -- the true "generation" transfer case -- and/or the
OBQA MCQA prompts), reads THREE aligned per-token series in a single forward pass:

    inf_log_var[t] : ||L_t||_F^2 = tr(posterior cov) of the FCVR router at
                     position t, averaged over the FCVR layers.  (the signal)
    entropy[t]     : Shannon entropy of the model's next-token distribution
                     softmax(LM_logits[t]) over the full vocab.  (reference UQ)
    surprisal[t]   : -log p(actual next token | context)          (reference UQ)

and then quantifies whether inf_log_var tracks entropy/surprisal:
  * correlation (Pearson + Spearman, WITH SIGN and p-value),
  * per-token-category elevation (numbers / entities / connectives vs "other"),
  * spike overlap (do the top-k% inf_log_var positions coincide with the
    top-k% entropy positions -- and, because the OoD signal is INVERTED on this
    checkpoint, also with the BOTTOM-k%).

Sign matters. On this checkpoint the sequence-level Inf-Logit-Var OoD signal is
INVERTED (higher variance on ID than OoD; matches the paper magnitude only after
1-AUROC). So a null or NEGATIVE token-level correlation is a legitimate -- and
expected -- outcome, and is itself the finding.

Read the printed "VERDICT" block and results/token_analysis/*.json.

Deterministic routing
---------------------
By default the FCVR routers are put in `deterministic_readout` mode: routing uses
the posterior mean (no S=35 MC sampling), so entropy/surprisal are reproducible
and fast. The Cholesky factor (hence inf_log_var) is computed deterministically
either way, so this does not change the signal -- only the routing noise.
"""

import argparse
import html
import json
import os
import re
import string

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import setup_environment, load_exp_dataset, multiple_choice_prompt_engineer
from model import load_peft_model_and_adapter, load_tokenizer
from evaluate_fcvr import prepare_model_fcvr  # reuse faithful reconstruction

# ---------------------------------------------------------------------------
# Free-form prose probes (the "does it transfer to generation" test). Chosen to
# be dense in numbers, named entities and discourse connectives.
# ---------------------------------------------------------------------------
BUILTIN_TEXTS = [
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall; however, it was almost demolished in 1909.",
    "Albert Einstein published the theory of general relativity in 1915, whereas his special relativity paper appeared in 1905.",
    "Water boils at 100 degrees Celsius at sea level, but at higher altitudes it boils lower because the air pressure decreases.",
    "The Amazon River discharges roughly 209,000 cubic meters of water per second, which exceeds the next seven largest rivers combined.",
    "Although the meeting was scheduled for 3 PM, the CEO arrived late; therefore, the presentation started at 4:15 instead.",
    "Marie Curie won two Nobel Prizes: one in Physics in 1903 and another in Chemistry in 1911.",
    "The company's revenue grew from 12 million dollars in 2019 to 87 million in 2023, so investors were optimistic.",
    "Mount Everest, located in the Himalayas, reaches 8,849 meters above sea level, making it the tallest mountain on Earth.",
    "Because the experiment failed three times, the researchers revised their hypothesis and consequently changed the temperature to 250 Kelvin.",
    "Shakespeare wrote Hamlet around 1600, but the play was not printed until 1603.",
    "The speed of light is approximately 299,792 kilometers per second, whereas sound travels at only 343 meters per second.",
    "Napoleon was defeated at Waterloo in 1815; nevertheless, his legal code still influences France today.",
    "The recipe requires 2 cups of flour, 3 eggs, and 150 grams of sugar, although you can substitute honey if needed.",
    "NASA launched the Voyager 1 probe in 1977, and it entered interstellar space in 2012.",
    "Tokyo has a population of about 14 million people, making it larger than New York, London, or Paris.",
]

# Strong discourse/logical connectives (deliberately excludes high-frequency
# function words like "and/or/but/so" which are weak, noisy signals).
DISCOURSE_CONNECTIVES = {
    "however", "therefore", "whereas", "although", "because", "consequently",
    "nevertheless", "moreover", "furthermore", "hence", "thus", "meanwhile",
    "accordingly", "conversely", "nonetheless", "despite", "instead",
    "otherwise", "since", "while", "yet", "though",
}
SENT_END = {".", "!", "?"}
_PUNCT = set(string.punctuation)


def parse_args():
    p = argparse.ArgumentParser(description="Step 1: per-token Inf-Logit-Var vs entropy/surprisal transfer test.")
    # --- model / weights (mirror evaluate_fcvr.py so the same run is reconstructed) ---
    p.add_argument("--model_shortcode", type=str, default="granite")
    p.add_argument("--dataset_shortcode", type=str, default="obqa",
                   help="ID dataset the FCVR model was trained on (drives weight paths).")
    p.add_argument("--kvq_adapter_path", type=str, required=True,
                   help="Stage-1 KVQ LoRA adapter path.")
    p.add_argument("--swap_layers", type=int, nargs="+", required=True,
                   help="FCVR layers to load (e.g. the Susceptible-10 set).")
    p.add_argument("--run_suffix", type=str, default=None,
                   help="Must match the training run's --run_suffix (FCVR weight dir).")
    p.add_argument("--prior_source", type=str, default="pretrained", choices=["map", "pretrained"],
                   help="Must match the training run.")
    p.add_argument("--num_samples", type=int, default=1,
                   help="MC samples if --stochastic_routing; ignored in the default deterministic mode.")
    p.add_argument("--stochastic_routing", action="store_true",
                   help="Route via S-sample MC (paper inference). Default: deterministic posterior-mean routing.")
    # --- what text to analyse ---
    p.add_argument("--source", type=str, default="builtin", choices=["builtin", "obqa", "textfile"],
                   help="builtin=free-form prose (transfer test); obqa=MCQA test prompts; textfile=one passage/line.")
    p.add_argument("--text_path", type=str, default=None, help="Required if --source textfile.")
    p.add_argument("--num_examples", type=int, default=50,
                   help="Max examples (obqa/textfile). builtin uses all 15.")
    p.add_argument("--chat_template", action="store_true",
                   help="Wrap builtin/textfile text in the MCQA chat template (matches training domain). "
                        "Default off for builtin (raw prose = true generation transfer). Always on for obqa.")
    # --- analysis knobs ---
    p.add_argument("--spike_pct", type=float, default=0.15, help="Top/bottom fraction defining a 'spike'.")
    p.add_argument("--max_html_examples", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="results/token_analysis")
    p.add_argument("--tag", type=str, default=None, help="Extra tag appended to output filenames.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Token categorisation (heuristic -- no spaCy dependency). Caveats:
#   * "number"    : token contains a digit.
#   * "connective": stripped/lowercased token in DISCOURSE_CONNECTIVES.
#   * "entity"    : capitalised alphabetic token NOT at sentence start (crude
#                   proper-noun proxy; will misfire on sentence-initial words --
#                   which is why sentence-initial capitals are excluded).
#   * "punct" / "other".
# These are coarse; they exist to see whether the signal is *elevated on average*
# on such tokens, not to be a NER system. The continuous correlation with
# entropy/surprisal is the more robust result.
# ---------------------------------------------------------------------------
def categorize(tok_str, prev_tok_str, is_first):
    s = tok_str.strip()
    if s == "":
        return "other"
    if re.search(r"\d", s):
        return "number"
    low = s.lower().strip(string.punctuation)
    if low in DISCOURSE_CONNECTIVES:
        return "connective"
    if all(ch in _PUNCT for ch in s):
        return "punct"
    prev = (prev_tok_str or "").strip()
    sent_initial = is_first or (prev != "" and prev[-1] in SENT_END)
    if s[0].isupper() and s.isalpha() and len(s) > 1 and not sent_initial:
        return "entity"
    return "other"


def build_texts(args, tokenizer):
    """Return a list of (raw_text_for_display, tokenized_string) pairs."""
    if args.source == "builtin":
        raw = BUILTIN_TEXTS
    elif args.source == "textfile":
        assert args.text_path, "--text_path required for --source textfile"
        with open(args.text_path) as f:
            raw = [ln.strip() for ln in f if ln.strip()][: args.num_examples]
    else:  # obqa (or any load_exp_dataset code)
        ds = load_exp_dataset(args.dataset_shortcode, split="test")[: args.num_examples]
        # Use the exact MCQA chat template the model was trained/evaluated with.
        return [(x["question"], multiple_choice_prompt_engineer(x, tokenizer=tokenizer)["question"]) for x in ds]

    use_template = args.chat_template
    out = []
    for t in raw:
        if use_template:
            templated = multiple_choice_prompt_engineer({"question": t, "answer": "", "id": ""}, tokenizer=tokenizer)["question"]
            out.append((t, templated))
        else:
            out.append((t, t))
    return out


@torch.no_grad()
def collect_per_token(model, tokenizer, text_string, fcvr_layers, causal_model, device):
    """One forward pass over a single sequence (batch_size=1, no padding).

    Returns a list of per-position dicts aligned so that everything at index t
    describes 'position t, about to predict token t+1':
      tok (input token t), next_tok (t+1), category (of next_tok),
      entropy[t], surprisal[t], inf_log_var[t], and per-layer inf_log_var.
    """
    inputs = tokenizer(text_string, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"][0]
    seq_len = input_ids.shape[0]
    if seq_len < 3:
        return []

    logits = model(**inputs).logits[0].float()  # [seq, vocab]

    # predictive entropy of next-token dist at each position
    logp = F.log_softmax(logits, dim=-1)
    probs = logp.exp()
    entropy = -(probs * logp).sum(dim=-1)  # [seq]
    # surprisal of the ACTUAL next token
    next_ids = input_ids[1:]
    surprisal = -logp[torch.arange(seq_len - 1, device=device), next_ids]  # [seq-1]

    # inf_log_var per position per FCVR layer
    per_layer = []
    for l in fcvr_layers:
        L = causal_model.layers[l].block_sparse_moe.router.last_cholesky_factor
        E = L.shape[-1]
        L = L.view(1, seq_len, E, E)[0].float()          # [seq, E, E]
        per_layer.append((L ** 2).sum(dim=(-1, -2)))      # [seq] = ||L||_F^2 = tr(LL^T)
    ilv_layers = torch.stack(per_layer, dim=0)            # [n_layers, seq]
    ilv = ilv_layers.mean(dim=0)                          # [seq]

    tok_strs = [tokenizer.decode([i]) for i in input_ids.tolist()]

    records = []
    for t in range(seq_len - 1):
        next_tok = tok_strs[t + 1]
        cat = categorize(next_tok, tok_strs[t], is_first=(t == 0))
        records.append({
            "t": t,
            "tok": tok_strs[t],
            "next_tok": next_tok,
            "category": cat,
            "entropy": float(entropy[t].item()),
            "surprisal": float(surprisal[t].item()),
            "inf_log_var": float(ilv[t].item()),
            "inf_log_var_per_layer": [float(v) for v in ilv_layers[:, t].tolist()],
        })
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return float(np.corrcoef(x, y)[0, 1]), float("nan")


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        if rx.std() == 0 or ry.std() == 0:
            return float("nan"), float("nan")
        return float(np.corrcoef(rx, ry)[0, 1]), float("nan")


def _zscore(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std() if v.std() > 0 else np.zeros_like(v)


def analyze(all_records, per_example_records, spike_pct):
    """all_records: flat list across every example. per_example_records: list of lists."""
    ent = [r["entropy"] for r in all_records]
    sur = [r["surprisal"] for r in all_records]
    ilv = [r["inf_log_var"] for r in all_records]

    # --- pooled raw correlations ---
    corr = {
        "pooled_raw": {
            "spearman_ilv_entropy": _spearman(ilv, ent),
            "spearman_ilv_surprisal": _spearman(ilv, sur),
            "pearson_ilv_entropy": _pearson(ilv, ent),
            "pearson_ilv_surprisal": _pearson(ilv, sur),
            "spearman_entropy_surprisal": _spearman(ent, sur),  # sanity: entropy vs surprisal
        }
    }
    # --- per-example z-scored, then pooled (removes between-text scale) ---
    zent, zilv, zsur = [], [], []
    per_ex_spearman = []
    for recs in per_example_records:
        if len(recs) < 3:
            continue
        e = _zscore([r["entropy"] for r in recs])
        i = _zscore([r["inf_log_var"] for r in recs])
        s = _zscore([r["surprisal"] for r in recs])
        zent += e.tolist(); zilv += i.tolist(); zsur += s.tolist()
        per_ex_spearman.append(_spearman([r["inf_log_var"] for r in recs], [r["entropy"] for r in recs])[0])
    corr["pooled_zscored"] = {
        "spearman_ilv_entropy": _spearman(zilv, zent),
        "spearman_ilv_surprisal": _spearman(zilv, zsur),
    }
    per_ex_spearman = [x for x in per_ex_spearman if not np.isnan(x)]
    corr["per_example_spearman_ilv_entropy"] = {
        "mean": float(np.mean(per_ex_spearman)) if per_ex_spearman else float("nan"),
        "std": float(np.std(per_ex_spearman)) if per_ex_spearman else float("nan"),
        "n": len(per_ex_spearman),
        "frac_positive": float(np.mean([x > 0 for x in per_ex_spearman])) if per_ex_spearman else float("nan"),
    }

    # --- per-category elevation (z-scored within example so scale is comparable) ---
    cat_z = {}
    for recs in per_example_records:
        if len(recs) < 3:
            continue
        zi = _zscore([r["inf_log_var"] for r in recs])
        ze = _zscore([r["entropy"] for r in recs])
        for k, r in enumerate(recs):
            cat_z.setdefault(r["category"], {"ilv_z": [], "ent_z": []})
            cat_z[r["category"]]["ilv_z"].append(zi[k])
            cat_z[r["category"]]["ent_z"].append(ze[k])
    category = {}
    for cat, d in sorted(cat_z.items()):
        category[cat] = {
            "n": len(d["ilv_z"]),
            "inf_log_var_z_mean": float(np.mean(d["ilv_z"])),   # >0 => signal above that example's mean on this cat
            "entropy_z_mean": float(np.mean(d["ent_z"])),       # reference: does entropy elevate here?
        }

    # --- spike overlap (within-example z-scored, pooled indices) ---
    def jacc(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if (a | b) else 0.0
    ent_spike, ilv_spike, ilv_trough = [], [], []
    offset = 0
    for recs in per_example_records:
        n = len(recs)
        if n < 3:
            offset += n; continue
        e = _zscore([r["entropy"] for r in recs])
        i = _zscore([r["inf_log_var"] for r in recs])
        k = max(1, int(round(spike_pct * n)))
        ent_top = set(np.argsort(e)[-k:] + offset)
        ilv_top = set(np.argsort(i)[-k:] + offset)
        ilv_bot = set(np.argsort(i)[:k] + offset)
        ent_spike += list(ent_top); ilv_spike += list(ilv_top); ilv_trough += list(ilv_bot)
        offset += n
    p = spike_pct
    spikes = {
        "spike_pct": spike_pct,
        "jaccard_entropySpike_vs_ilvSpike": jacc(ent_spike, ilv_spike),
        "jaccard_entropySpike_vs_ilvTrough": jacc(ent_spike, ilv_trough),
        "random_baseline_jaccard": p / (2 - p) if p < 1 else 1.0,
        "note": "If ilvTrough overlap >> ilvSpike overlap, the token-level signal is INVERTED "
                "(entropy peaks coincide with LOW variance), consistent with the OoD inversion.",
    }
    return {"correlation": corr, "category": category, "spikes": spikes}


def verdict(summary):
    """Human-readable interpretation."""
    r = summary["correlation"]["per_example_spearman_ilv_entropy"]["mean"]
    zr = summary["correlation"]["pooled_zscored"]["spearman_ilv_entropy"][0]
    sp = summary["spikes"]
    lines = []
    lines.append(f"per-example mean Spearman(inf_log_var, entropy) = {r:+.3f}")
    lines.append(f"pooled z-scored Spearman(inf_log_var, entropy)  = {zr:+.3f}")
    lines.append(f"spike Jaccard  (entropy-top vs ilv-top)   = {sp['jaccard_entropySpike_vs_ilvSpike']:.3f}")
    lines.append(f"spike Jaccard  (entropy-top vs ilv-BOTTOM) = {sp['jaccard_entropySpike_vs_ilvTrough']:.3f}  "
                 f"(random ~ {sp['random_baseline_jaccard']:.3f})")
    ref = r if not np.isnan(r) else zr
    if np.isnan(ref):
        tag = "INCONCLUSIVE (not enough signal variance)"
    elif ref >= 0.20:
        tag = "TRANSFERS (positive) -> signal spikes with uncertainty; the decoding plan's premise holds."
    elif ref <= -0.20:
        tag = ("INVERTED at token level -> Inf-Logit-Var is LOW where entropy is HIGH. "
               "Consistent with the OoD sign inversion. Use it as an inverted/confidence signal, "
               "or treat this as the negative finding (flaw #5).")
    else:
        tag = ("NO TRANSFER (~0) -> the MCQA-trained router variance does not track per-token "
               "generation uncertainty. This is itself a valid finding (flaw #5).")
    lines.append("VERDICT: " + tag)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _heat(val, vmin, vmax):
    if vmax <= vmin:
        f = 0.0
    else:
        f = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    r, g, b = 255, int(255 * (1 - f)), int(255 * (1 - f))  # white -> red
    return f"rgb({r},{g},{b})"


def write_html(per_example_records, raw_texts, path, max_examples):
    css = ("body{font-family:monospace;background:#111;color:#eee;padding:16px;}"
           ".tok{padding:1px 2px;border-radius:2px;color:#000;}"
           ".lbl{color:#7fd;margin:14px 0 4px;font-weight:bold;}"
           ".cat{outline:2px solid #06f;}")
    parts = [f"<!doctype html><meta charset='utf-8'><style>{css}</style>",
             "<h2>Step 1: per-token signals (top row = predictive ENTROPY, bottom row = INF-LOG-VAR)</h2>",
             "<p>Redder = higher. Blue outline = number/entity/connective (the 'important' tokens). "
             "Compare whether the same tokens light up in both rows.</p>"]
    for ex_i, recs in enumerate(per_example_records[:max_examples]):
        if not recs:
            continue
        ents = [r["entropy"] for r in recs]; ilvs = [r["inf_log_var"] for r in recs]
        e_lo, e_hi = min(ents), max(ents); i_lo, i_hi = min(ilvs), max(ilvs)
        parts.append(f"<div class='lbl'>[{ex_i}] {html.escape(raw_texts[ex_i][:160])}</div>")
        for signal, lo, hi, key in (("ENTROPY", e_lo, e_hi, "entropy"), ("INF-LOG-VAR", i_lo, i_hi, "inf_log_var")):
            row = [f"<div><small style='color:#888'>{signal}</small><br>"]
            for r in recs:
                cls = "tok cat" if r["category"] in ("number", "entity", "connective") else "tok"
                bg = _heat(r[key], lo, hi)
                # display the predicted (next) token, which is what the signal at t is 'about'
                txt = html.escape(r["next_tok"]).replace(" ", "&nbsp;") or "&nbsp;"
                row.append(f"<span class='{cls}' style='background:{bg}' "
                           f"title='{r['category']} e={r['entropy']:.2f} ilv={r['inf_log_var']:.2f}'>{txt}</span>")
            row.append("</div>")
            parts.append("".join(row))
    with open(path, "w") as f:
        f.write("".join(parts))


def write_plots(all_records, summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable: {e}; skipping PNG)")
        return
    ent = np.array([r["entropy"] for r in all_records])
    ilv = np.array([r["inf_log_var"] for r in all_records])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(ent, ilv, s=6, alpha=0.3)
    sr = summary["correlation"]["pooled_raw"]["spearman_ilv_entropy"][0]
    ax[0].set_xlabel("predictive entropy"); ax[0].set_ylabel("inf_log_var")
    ax[0].set_title(f"per-token (pooled)  Spearman={sr:+.3f}")
    cats = summary["category"]
    order = [c for c in ["number", "entity", "connective", "other", "punct"] if c in cats]
    x = np.arange(len(order)); w = 0.38
    ax[1].bar(x - w / 2, [cats[c]["inf_log_var_z_mean"] for c in order], w, label="inf_log_var (z)")
    ax[1].bar(x + w / 2, [cats[c]["entropy_z_mean"] for c in order], w, label="entropy (z)")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(order); ax[1].legend()
    ax[1].set_title("mean z-scored signal by token category")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    setup_environment()
    args = parse_args()
    torch.manual_seed(args.seed)

    model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=args.kvq_adapter_path, device_map="cuda:0")
    tokenizer = load_tokenizer(args.model_shortcode)
    model = prepare_model_fcvr(model, args)  # loads MAP (if map-prior) + FCVR weights, sets num_mc_samples

    fcvr_layers = sorted(args.swap_layers)
    causal_model = model.base_model.model.model
    # deterministic posterior-mean routing unless explicitly asked for MC sampling
    det = not args.stochastic_routing
    for l in fcvr_layers:
        causal_model.layers[l].block_sparse_moe.router.deterministic_readout = det
    print(f"Routing mode: {'DETERMINISTIC posterior-mean' if det else f'STOCHASTIC (S={args.num_samples})'}")

    texts = build_texts(args, tokenizer)
    print(f"Analysing {len(texts)} examples from source='{args.source}' "
          f"(chat_template={'yes' if (args.chat_template or args.source == 'obqa') else 'no'})")

    per_example, raw_texts = [], []
    for raw, tokstr in tqdm(texts, desc="forward"):
        recs = collect_per_token(model, tokenizer, tokstr, fcvr_layers, causal_model, model.device)
        per_example.append(recs); raw_texts.append(raw)
    all_records = [r for recs in per_example for r in recs]
    print(f"Collected {len(all_records)} token positions.")

    summary = analyze(all_records, per_example, args.spike_pct)
    summary["config"] = {
        "source": args.source, "num_examples": len(texts), "n_tokens": len(all_records),
        "fcvr_layers": fcvr_layers, "prior_source": args.prior_source, "run_suffix": args.run_suffix,
        "routing": "deterministic" if det else f"stochastic_S{args.num_samples}",
        "chat_template": bool(args.chat_template or args.source == "obqa"),
    }
    v = verdict(summary)
    summary["verdict"] = v

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    base = os.path.join(args.output_dir, f"step1_{args.source}{tag}")
    with open(base + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    # per-token dump (drop the heavy per-layer list to keep it small)
    with open(base + "_pertoken.jsonl", "w") as f:
        for ex_i, recs in enumerate(per_example):
            for r in recs:
                rr = {k: val for k, val in r.items() if k != "inf_log_var_per_layer"}
                rr["example"] = ex_i
                f.write(json.dumps(rr) + "\n")
    write_html(per_example, raw_texts, base + ".html", args.max_html_examples)
    write_plots(all_records, summary, base + ".png")

    print("\n" + "=" * 70)
    print(v)
    print("=" * 70)
    print("\nPer-category (z-scored within example; >0 = elevated):")
    for c, d in summary["category"].items():
        print(f"  {c:11s} n={d['n']:5d}  ilv_z={d['inf_log_var_z_mean']:+.3f}  ent_z={d['entropy_z_mean']:+.3f}")
    print(f"\nSaved: {base}.json / _pertoken.jsonl / .html / .png")


if __name__ == "__main__":
    main()

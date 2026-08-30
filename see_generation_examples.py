"""Show what arm A (answer-only) and arm B (answer + fact1) actually GENERATE on
every dataset of the OBQA-comparison OoD read-out, ~20 examples per dataset,
formatted into one human-readable text file (branch OBQA-comparison).

Two generation modes per example (both by default, --modes to pick):

  free    greedy continuation of the arm's OWN chat prompt (its training system
          prompt: mcq for arm A, comparison for arm B), nothing forced -> what
          the arm says when left alone (arm A is expected to emit a bare letter,
          arm B "<letter>\\nExplanation: ...").
  forced  the evaluate_ood_expl_readout.py --stage gen protocol: prompt-only
          forward -> predicted letter (masked over the example's valid choices)
          -> forced prefix "<pred>\\nExplanation:" -> greedy decode (KV cache,
          <= --max_new_tokens or EOS). Rows, per-example MC seeds and decode
          kwargs are the ones of that stage, so the forced texts are the ones
          the OoD read-out scores (up to CUDA nondeterminism under stochastic
          routing).

Rows: the first --n_examples usable rows per dataset in the eval's own order
(explanation-bearing sets go through evaluate_ood_expl_readout.trace_rows_for_domain,
i.e. the same funnel as the tf/gen stages; letter-only sets such as arc_c/arc_e
take the first rows with a valid gold letter). Both arms see identical rows.

READ-ONLY w.r.t. every trained artefact. Outputs (refuse to overwrite unless
--overwrite), under --output_dir:
    {tag}_{split}_data-s{data_seed}_mc-s{sampling_seed}_{routing}.txt    the report
    {tag}_{split}_data-s{data_seed}_mc-s{sampling_seed}_{routing}.jsonl  one record per (dataset, example, arm)

Usage (quail-1, repo root, moe_env; ~30-60 min for 7 datasets x 20 x 2 modes x 2 arms,
arm A's forced explanations tend to run to the token cap):

    python see_generation_examples.py
    python see_generation_examples.py --n_examples 10 --max_new_tokens 128 --tag quick
    python see_generation_examples.py --routing deterministic            # posterior-mean routing
    python see_generation_examples.py --modes free                       # only the unforced continuations
    python see_generation_examples.py --datasets obqa_gen medexqa ecqa   # subset

No GPU needed to re-format the texts a finished --stage gen run already saved
(forced mode only; the question text is re-loaded from the datasets):

    python see_generation_examples.py --from_texts results/ood_expl_readout/obqa-ood-expl_test_data-s42_mc-s42_gen_texts.jsonl
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import textwrap
import time
from collections import Counter, OrderedDict

import torch
from tqdm import tqdm

from analyze_obqa_gen import cell_name
from evaluate_ilv_ood_arms import (
    ARM_SETUP,
    DEFAULT_LAYERS,
    SAME_SOURCE,
    check_output_collisions,
    git_revision,
    load_domains,
    resolve_arm_setup,
)
from evaluate_ood_expl_readout import (
    ARM_KEYS,
    DOMAIN_SEED_INDEX,
    LETTERS,
    arm_prompt,
    gold_letter_of,
    letter_probe,
    letter_token_ids,
    load_arm,
    marker_token_ids,
    n_choices_of,
    release_arm,
    trace_rows_for_domain,
)
from model import load_tokenizer
from utils import setup_environment
from utils.data import EXPLANATION_MARKER
from utils.prompt import SYSTEM_INSTRUCTIONS

DEFAULT_DATASETS = ["obqa_gen", "arc_c", "arc_e", "medexqa", "scienceqa", "ecqa", "aqua_rat"]
MODES = ("free", "forced")
ARM_TITLE = {"armA-letter": "ARM A (answer-only)", "armB-ansexp": "ARM B (answer + explanation)"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                   help="Datasets to show (ID obqa_gen + the stage1/tf/gen OoD sets by default).")
    p.add_argument("--n_examples", type=int, default=20, help="Examples per dataset (0 = every loaded row).")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--arms", nargs="+", choices=ARM_KEYS, default=list(ARM_KEYS))
    p.add_argument("--model_shortcode", default="granite")
    p.add_argument("--id_dataset", default="obqa_gen", choices=["obqa_gen"])
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--n_per_domain", type=int, default=500,
                   help="Rows loaded per domain before selection (eval default 500; 0 = all).")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_seq_tokens", type=int, default=2048, help="Trace-funnel cap (eval default).")
    p.add_argument("--data_seed", type=int, default=42, help="ONLY dataset construction/split seed.")
    p.add_argument("--sampling_seed", type=int, default=42, help="ONLY FCVR Monte Carlo seed.")
    p.add_argument("--num_samples", type=int, default=35, help="FCVR MC samples S (paper: 35).")
    p.add_argument("--batch_size", type=int, default=8, help="Only forwarded to the model loader.")
    p.add_argument("--routing", choices=["stochastic", "deterministic"], default="stochastic",
                   help="stochastic = eval protocol (seeded MC routing); deterministic = posterior-mean routing.")
    p.add_argument("--swap_layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--prior_source", choices=["pretrained", "map"], default="pretrained")
    p.add_argument("--map_suffix", default=None)
    p.add_argument("--arm_a_adapter", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--arm_b_adapter", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--arm_a_run_suffix", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--arm_b_run_suffix", default=None, help="Default: ARM_SETUP entry.")
    p.add_argument("--output_dir", default="results/generation_examples")
    p.add_argument("--tag", default="gen-examples")
    p.add_argument("--width", type=int, default=100, help="Report line width.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--from_texts", default=None, metavar="GEN_TEXTS_JSONL",
                   help="Format an existing --stage gen *_texts.jsonl instead of generating (CPU only).")
    return p.parse_args(argv)


def validate_args(args):
    if args.model_shortcode != "granite":
        raise SystemExit("The saved comparison arms are Granite-specific; --model_shortcode must be granite.")
    if args.n_examples < 0 or args.n_per_domain < 0 or args.max_new_tokens < 1 or args.width < 40:
        raise SystemExit("--n_examples/--n_per_domain >= 0, --max_new_tokens >= 1, --width >= 40 required")
    if not args.swap_layers:
        raise SystemExit("--swap_layers must contain the FCVR-modified layers")
    seen = []
    for code in args.datasets:
        if code in SAME_SOURCE.get(args.id_dataset, set()):
            raise SystemExit(f"ERROR: {code} overlaps the {args.id_dataset} training corpus (use {args.id_dataset}).")
        if code not in DOMAIN_SEED_INDEX:
            raise SystemExit(f"ERROR: {code} has no DOMAIN_SEED_INDEX entry in evaluate_ood_expl_readout.py.")
        if code not in seen:
            seen.append(code)
    args.datasets = seen
    # load_domains() always loads the ID anchor plus these.
    args.ood_datasets = [c for c in seen if c != args.id_dataset]


def output_paths(args):
    stem = os.path.join(args.output_dir,
                        f"{args.tag}_{args.split}_data-s{args.data_seed}_mc-s{args.sampling_seed}_{args.routing}")
    return {"txt": stem + ".txt", "jsonl": stem + ".jsonl"}


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------
def gold_explanation_of(ex):
    """Loader convention: explanation-bearing rows carry letter_question +
    gold_letter and `answer` = ' ' + explanation; letter-only MCQA rows keep the
    letter in `answer` (no explanation)."""
    if ex.get("letter_question") and ex.get("gold_letter"):
        s = str(ex.get("answer") or "").strip()
        return s or None
    return None


def question_text_of(ex):
    return ex.get("letter_question") or ex.get("question") or ""


def select_rows(d, n, tokenizer, marker_ids, max_seq_tokens):
    """First n usable rows in eval order. Explanation-bearing sets: the tf/gen
    trace funnel (identical rows + ex_index -> identical MC seeds). Letter-only
    sets: first rows with a valid gold letter."""
    rows, ids, funnel = trace_rows_for_domain(d, n, tokenizer, marker_ids, max_seq_tokens)
    if rows:
        return rows, ids, "eval trace funnel (same rows as --stage tf/gen)", funnel
    rows, ids, funnel = [], [], Counter()
    for i, ex in enumerate(d.rows):
        if gold_letter_of(ex) is None or not question_text_of(ex):
            funnel["no_letter_or_question"] += 1
            continue
        rows.append(ex); ids.append(d.ids[i]); funnel["kept"] += 1
        if n and len(rows) >= n:
            break
    return rows, ids, "first rows with a valid gold letter (letter-only set)", dict(funnel)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def example_seed(args, code, ex_i):
    """evaluate_ood_expl_readout.run_trace_arm seed formula."""
    return args.sampling_seed * 100003 + DOMAIN_SEED_INDEX[code] * 100000 + ex_i


@torch.no_grad()
def greedy_generate(model, tokenizer, prefix_ids, max_new_tokens, seed):
    """Same decode kwargs / seeding as evaluate_ood_expl_readout.generate_explanation
    (minus the per-step logits)."""
    device = model.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inp = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new_tokens,
                         do_sample=False, num_beams=1, use_cache=True, eos_token_id=eos_id, pad_token_id=pad_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_ids = [int(x) for x in out[0, len(prefix_ids):].tolist()]
    G = len(gen_ids)
    ended = bool(G > 0 and gen_ids[-1] == eos_id)
    return {"text": tokenizer.decode(gen_ids, skip_special_tokens=True), "n_tokens": G,
            "ended_with_eos": ended, "truncated": bool(not ended and G >= max_new_tokens),
            "seconds": time.perf_counter() - t0}


def free_format_flags(text):
    """Does an unforced continuation look like the trained target?"""
    s = text.lstrip()
    first = s[:1]
    return {"first_letter": first if first in LETTERS else None,
            "has_marker": EXPLANATION_MARKER.strip() in text}


def generate_arm(model, tokenizer, args, arm_key, cfg, selected, fh):
    """All datasets for one loaded arm. Streams one record per example."""
    causal_model = model.base_model.model.model
    system_instruction = SYSTEM_INSTRUCTIONS[cfg["system_prompt"]]
    cell = cell_name(arm_key, cfg["system_prompt"])
    marker_ids = marker_token_ids(tokenizer)
    letter_ids = letter_token_ids(tokenizer)
    choice_ids_t = torch.tensor(letter_ids, device=model.device)
    fcvr_layers = sorted(args.swap_layers)
    records = []
    for code, (rows, ids, _, _) in selected.items():
        for ex_i, (ex, ex_id) in enumerate(tqdm(list(zip(rows, ids)), desc=f"{cell} {code}")):
            seed = example_seed(args, code, ex_i)
            gold, nch = gold_letter_of(ex), n_choices_of(ex)
            prompt = arm_prompt(tokenizer, ex, system_instruction)
            prompt_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
            rec = {"arm": arm_key, "cell": cell, "system_prompt": cfg["system_prompt"], "dataset": code,
                   "is_ood": code != args.id_dataset, "id": ex_id, "ex_index": ex_i, "mc_seed": seed,
                   "gold_letter": gold, "n_choices": nch, "n_prompt_tokens": len(prompt_ids)}
            if "free" in args.modes:
                g = greedy_generate(model, tokenizer, prompt_ids, args.max_new_tokens, seed)
                rec["free"] = {**g, **free_format_flags(g["text"])}
            if "forced" in args.modes:
                probe = letter_probe(model, causal_model, fcvr_layers, prompt_ids, choice_ids_t, nch, seed)
                pred = probe["pred_letter"]
                prefix = prompt_ids + [letter_ids[LETTERS.index(pred)]] + marker_ids
                g = greedy_generate(model, tokenizer, prefix, args.max_new_tokens, seed)
                rec["forced"] = {"pred_letter": pred, "pred_prob": float(probe["probs"].max()),
                                 "probs": {L: float(p) for L, p in zip(LETTERS[:nch], probe["probs"][:nch])},
                                 "correct": bool(pred == gold), "letter_entropy": float(probe["letter_entropy"]),
                                 "prompt_final_ilv": float(probe["prompt_final_ilv_promptonly"]), **g}
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
    return records


# ---------------------------------------------------------------------------
# --from_texts: re-format a finished --stage gen texts file (forced mode only)
# ---------------------------------------------------------------------------
def records_from_texts(path, args, domains, fh):
    """gen *_texts.jsonl rows -> records; keeps the first n ids per dataset in
    file order (the eval's trace order), both arms on the same ids."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    id_to_row = {code: dict(zip(d.ids, d.rows)) for code, d in domains.items()}
    keep_ids = OrderedDict()
    for r in rows:
        if r["dataset"] not in args.datasets:
            continue
        lst = keep_ids.setdefault(r["dataset"], [])
        if r["id"] not in lst and (not args.n_examples or len(lst) < args.n_examples):
            lst.append(r["id"])
    def row_of(code, ex_id):
        ex = id_to_row.get(code, {}).get(ex_id)
        if ex is None:                        # ids are per-load; reload with the eval's --split/--n_per_domain
            ex = {"question": f"<row {ex_id} not found in the reloaded {code}/{args.split} rows; "
                              f"rerun with the eval's --split/--n_per_domain>", "n_choices": None}
        return ex

    selected = {}
    for code, ids in keep_ids.items():
        selected[code] = ([row_of(code, i) for i in ids], ids, f"first ids of {os.path.basename(path)}", {"kept": len(ids)})
    records = []
    for r in rows:
        code = r["dataset"]
        if code not in keep_ids or r["id"] not in keep_ids[code] or r["arm"] not in args.arms:
            continue
        ex = row_of(code, r["id"])
        rec = {"arm": r["arm"], "cell": r.get("cell"), "system_prompt": None, "dataset": code,
               "is_ood": r.get("is_ood"), "id": r["id"], "ex_index": keep_ids[code].index(r["id"]), "mc_seed": None,
               "gold_letter": r.get("gold_letter"), "n_choices": r.get("n_choices") or n_choices_of(ex),
               "forced": {"pred_letter": r["pred_letter"], "pred_prob": None, "probs": None,
                          "correct": bool(r["pred_letter"] == r.get("gold_letter")), "text": r["gen_text"],
                          "n_tokens": r.get("n_gen_tokens"), "ended_with_eos": r.get("ended_with_eos"),
                          "truncated": r.get("truncated"), "seconds": None}}
        records.append(rec)
        fh.write(json.dumps(rec) + "\n")
    return selected, records


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def wrap_block(text, width, indent="    "):
    """Wrap each line separately so the question/option line structure survives."""
    out = []
    for line in (text if text else "").split("\n"):
        if not line.strip():
            out.append(indent.rstrip())
            continue
        out.extend(textwrap.wrap(line, width=width, initial_indent=indent, subsequent_indent=indent,
                                 break_long_words=False, break_on_hyphens=False) or [indent.rstrip()])
    return "\n".join(out)


def _fmt_gen_status(g):
    if g is None:
        return "n/a"
    n = g.get("n_tokens")
    n_s = f"{n} tok" if n is not None else "? tok"
    if g.get("truncated"):
        return f"{n_s}, TRUNCATED at cap"
    return f"{n_s}, {'eos' if g.get('ended_with_eos') else 'no eos'}"


def _pct(num, den):
    return f"{100.0 * num / den:5.1f}%" if den else "  n/a "


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return f"{sum(vals) / len(vals):6.1f}" if vals else "   n/a"


def summary_table(records, arms, datasets):
    by = {}
    for r in records:
        by.setdefault((r["dataset"], r["arm"]), []).append(r)
    hdr = (f"{'dataset':<10} {'arm':<5} {'n':>3} | {'forced acc':>10} {'forced len':>10} {'forced trunc':>12} | "
           f"{'free letter-1st':>15} {'free has Expl:':>14} {'free len':>8} {'free trunc':>10}")
    lines = [hdr, "-" * len(hdr)]
    for code in datasets:
        for arm in arms:
            rs = by.get((code, arm), [])
            if not rs:
                continue
            fo = [r["forced"] for r in rs if r.get("forced")]
            fr = [r["free"] for r in rs if r.get("free")]
            lines.append(
                f"{code:<10} {arm.split('-')[0]:<5} {len(rs):>3} | "
                f"{_pct(sum(g['correct'] for g in fo), len(fo)):>10} {_mean([g.get('n_tokens') for g in fo]):>10} "
                f"{_pct(sum(bool(g.get('truncated')) for g in fo), len(fo)):>12} | "
                f"{_pct(sum(g['first_letter'] is not None for g in fr), len(fr)):>15} "
                f"{_pct(sum(g['has_marker'] for g in fr), len(fr)):>14} {_mean([g.get('n_tokens') for g in fr]):>8} "
                f"{_pct(sum(bool(g.get('truncated')) for g in fr), len(fr)):>10}")
    return "\n".join(lines)


def write_report(path, args, arm_cfgs, setup, selected, records, source_note):
    W = args.width
    idx = {(r["dataset"], r["id"], r["arm"]): r for r in records}
    ood_desc = setup.get("ood_description", {})
    L = []
    L.append("=" * W)
    L.append("GENERATION EXAMPLES -- arm A (answer-only) vs arm B (answer + explanation), ID = obqa_gen")
    L.append("=" * W)
    L.append(f"written {_dt.datetime.now().isoformat(timespec='seconds')}  git={git_revision()}  source={source_note}")
    L.append(f"split={args.split}  routing={args.routing}  S={args.num_samples}  max_new_tokens={args.max_new_tokens}  "
             f"data_seed={args.data_seed}  sampling_seed={args.sampling_seed}  layers={sorted(args.swap_layers)}")
    for k in args.arms:
        c = arm_cfgs[k]
        L.append(f"{k:<12} {c['label']}")
        L.append(f"{'':<12} adapter={c['adapter']}  fcvr_suffix={c['run_suffix']}  "
                 f"weights_ds={c['weights_dataset'] or args.id_dataset}  system_prompt={c['system_prompt']}")
    L.append("")
    L.append("modes")
    L.append("  [free]   greedy continuation of the arm's own chat prompt, nothing forced")
    L.append("  [forced] eval protocol (evaluate_ood_expl_readout.py --stage gen): predicted letter (masked over valid")
    L.append("           choices) -> forced \"<pred>\\nExplanation:\" -> greedy decode; the text shown is what follows the prefix")
    L.append("")
    L.append("datasets")
    for code, (rows, _, sel, funnel) in selected.items():
        tag = "ID" if code == args.id_dataset else "OoD: " + ood_desc.get(code, "unclassified shift")
        nch = n_choices_of(rows[0]) if rows else "?"
        L.append(f"  {code:<10} {len(rows):>3} examples  n_choices={nch}  [{tag}]  selection: {sel}  funnel={funnel}")
    L.append("")
    L.append("SUMMARY (per dataset x arm; 'forced acc' = predicted-letter accuracy of the prompt-only probe; "
             "'free letter-1st' = unforced text starts with a choice letter)")
    L.append(summary_table(records, args.arms, list(selected)))
    L.append("")

    for code, (rows, ids, _, _) in selected.items():
        tag = "ID" if code == args.id_dataset else "OoD -- " + ood_desc.get(code, "unclassified shift")
        L.append("#" * W)
        L.append(f"DATASET {code}  ({tag}; {len(rows)} examples)")
        L.append("#" * W)
        for i, (ex, ex_id) in enumerate(zip(rows, ids)):
            gold = gold_letter_of(ex)
            L.append("")
            L.append(f"--- [{code} {i + 1}/{len(rows)}]  id={ex_id}  gold={gold}  n_choices={n_choices_of(ex)} ---")
            L.append("QUESTION")
            L.append(wrap_block(question_text_of(ex), W))
            gexp = gold_explanation_of(ex)
            if gexp:
                L.append("GOLD EXPLANATION" + (" (fact1)" if code == "obqa_gen" else ""))
                L.append(wrap_block(gexp, W))
            for arm in args.arms:
                r = idx.get((code, ex_id, arm))
                sp = arm_cfgs[arm]["system_prompt"]
                L.append(f"{ARM_TITLE[arm]}  [system_prompt={sp}]")
                if r is None:
                    L.append("    (no record)")
                    continue
                g = r.get("free")
                if g is not None:
                    L.append(f"  [free]   ({_fmt_gen_status(g)})")
                    L.append(wrap_block(g["text"] if g["text"].strip() else "<empty>", W, indent="    | "))
                g = r.get("forced")
                if g is not None:
                    p = f", p={g['pred_prob']:.2f}" if g.get("pred_prob") is not None else ""
                    verdict = "correct" if g["correct"] else f"WRONG (gold {gold})"
                    L.append(f"  [forced] pred={g['pred_letter']}{p} {verdict}; prefix \"{g['pred_letter']}\\nExplanation:\" "
                             f"then ({_fmt_gen_status(g)})")
                    L.append(wrap_block(g["text"] if g["text"].strip() else "<empty>", W, indent="    | "))
        L.append("")
    L.append("=" * W)
    L.append("end")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    validate_args(args)                       # sets args.ood_datasets before resolve_arm_setup reads it
    setup, arm_cfgs = resolve_arm_setup(args)
    paths = output_paths(args)
    check_output_collisions(list(paths.values()), args.overwrite)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_environment()

    print("#" * 80)
    print(f"# generation examples | datasets={args.datasets} n={args.n_examples} modes={args.modes} arms={args.arms}")
    print(f"# split={args.split} routing={args.routing} S={args.num_samples} max_new_tokens={args.max_new_tokens} "
          f"data_seed={args.data_seed} sampling_seed={args.sampling_seed}")
    for k in args.arms:
        c = arm_cfgs[k]
        print(f"# {k}: adapter={c['adapter']} suffix={c['run_suffix']} prompt={c['system_prompt']}")
    print(f"# outputs: {paths['txt']} | {paths['jsonl']}")
    print("#" * 80)

    domains = load_domains(args)

    if args.from_texts:
        with open(paths["jsonl"], "w", encoding="utf-8") as fh:
            selected, records = records_from_texts(args.from_texts, args, domains, fh)
        if not records:
            raise SystemExit(f"no rows for datasets {args.datasets} / arms {args.arms} in {args.from_texts}")
        write_report(paths["txt"], args, arm_cfgs, setup, selected, records, f"from_texts {args.from_texts}")
        print(f"wrote {paths['txt']} ({len(records)} records)")
        return

    tok0 = load_tokenizer(args.model_shortcode)
    marker_ids = marker_token_ids(tok0)
    selected = OrderedDict()
    for code in args.datasets:
        rows, ids, sel, funnel = select_rows(domains[code], args.n_examples, tok0, marker_ids, args.max_seq_tokens)
        if not rows:
            raise SystemExit(f"{code}: no usable rows ({funnel})")
        selected[code] = (rows, ids, sel, funnel)
        print(f"{code}: {len(rows)} rows selected -- {sel}; funnel={funnel}")

    records = []
    with open(paths["jsonl"], "w", encoding="utf-8") as fh:
        for arm_key in args.arms:
            cfg = arm_cfgs[arm_key]
            print("\n" + "=" * 80 + f"\n{cfg['label']}\n" + "=" * 80)
            model, tokenizer, _ = load_arm(args, cfg)
            try:
                records += generate_arm(model, tokenizer, args, arm_key, cfg, selected, fh)
            finally:
                release_arm(model)
            # Keep the report usable after a crash in the second arm.
            write_report(paths["txt"], args, arm_cfgs, setup, selected, records, "live generation (partial)")
    write_report(paths["txt"], args, arm_cfgs, setup, selected, records, "live generation")
    print(f"\nwrote {paths['txt']} ({len(records)} records; jsonl {paths['jsonl']})")


if __name__ == "__main__":
    main()

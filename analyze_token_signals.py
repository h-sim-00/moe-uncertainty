"""
Step 1 (new-exp2) -- Does the FCVR (VGLR-FC) Inf-Logit-Var router signal transfer
to per-token GENERATION, now that the router heads were fine-tuned on an
open-generation dataset (MedExQA) instead of single-token MCQA?

On new-exp1 this same probe ran on an OBQA(MCQA)-trained checkpoint over prose --
so any per-token structure was pure transfer from a letter-scoring objective. On
new-exp2 the Granite+FCVR routers are trained on MedExQA free-text explanations,
so this script measures whether Inf-Logit-Var tracks token-level uncertainty when
the heads actually saw generation during training.

It reads THREE aligned per-token series in a single forward pass:

    inf_log_var[t] : ||L_t||_F^2 = tr(posterior cov) of the FCVR router at
                     position t, averaged over the FCVR layers.  (the signal)
    entropy[t]     : Shannon entropy of softmax(LM_logits[t]) over the full
                     vocab.                                       (reference UQ)
    surprisal[t]   : -log p(actual next token | context).         (reference UQ)

and quantifies whether inf_log_var tracks entropy/surprisal (Pearson + Spearman
with sign, per-token-category elevation, spike overlap).

Two readouts on the MedExQA source (run BOTH; see fcvr-eval-granite-medexqa.sh):
  * teacher-forced (default): the sequence is prompt + GOLD explanation, so the
    per-token signal is read over the reference answer.
  * --generate: the model greedily generates its own explanation; the signal is
    read over the MODEL'S generated tokens (the realistic decoding case).
For the MedExQA source only the ANSWER region (explanation tokens) is analysed --
the prompt tokens are excluded because they are not what a decoder acts on.

Sources 'builtin' (free prose) and 'obqa' (MCQA prompts) are kept for parity with
new-exp1 and analyse every position.

Read the printed "VERDICT" block and results/token_analysis/*.json.

Routing at readout (--routing)
------------------------------
Default `stochastic` = the paper's inference: S (=35) samples from the logit
posterior, softmax-averaged, then Top-K. `deterministic` routes on the posterior
mean instead. NOTE: deterministic routing is an ABLATION, not a free
simplification -- expert choices in earlier FCVR layers change the hidden states
seen by later layers, hence later covariances, entropies and (with --generate) the
whole generated trajectory. Only the per-layer L given a fixed input is unaffected.
The MC sampling is in logit space and the experts run once, so S=35 costs ~nothing.

Online vs post-hoc ILV (--generate)
-----------------------------------
`ilv_online` is captured DURING generation with forward hooks on the FCVR routers:
for generated token t it is tr(L L^T) at the position that produced token t, in
the very forward pass (and, under stochastic routing, the very posterior draw) that
routed it -- a decoding-time score. `ilv_posthoc` re-reads the finished sequence in
a second teacher-forced pass (the old readout); under stochastic routing it is a
different draw and, in general, different hidden states. Both are written; the
online series is PRIMARY (all `inf_log_var`/`entropy`/`surprisal` per-token fields
and the `ilv_online_*` sequence aggregates), the post-hoc one is reported as
`*_posthoc` for comparison.
"""

import argparse
import html
import json
import os
import re
import string
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import (
    setup_environment,
    load_exp_dataset,
    multiple_choice_prompt_engineer,
    generation_prompt_engineer,
)
from model import load_peft_model_and_adapter, load_tokenizer
from evaluate_fcvr import prepare_model_by_arm  # reuse faithful reconstruction (+ baseline arms)

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
    p.add_argument("--dataset_shortcode", type=str, default="medexqa",
                   help="ID dataset the FCVR model was trained on (drives weight paths).")
    p.add_argument("--kvq_adapter_path", type=str, required=True,
                   help="Stage-1 KVQ LoRA adapter path.")
    p.add_argument("--swap_layers", type=int, nargs="+", default=[],
                   help="FCVR layers to load (e.g. the Susceptible-10 set). Required unless --arm det.")
    p.add_argument("--arm", type=str, default="fcvr", choices=["fcvr", "untrained", "det"],
                   help="Baseline ladder: fcvr = trained FCVR weights (default); untrained = fresh FCVR "
                        "heads on the swap layers (no Stage-2 training); det = Stage-1 model with stock "
                        "deterministic routers (no ILV; entropy/NLL/length baselines + generation quality only).")
    p.add_argument("--run_suffix", type=str, default=None,
                   help="Must match the training run's --run_suffix (FCVR weight dir).")
    p.add_argument("--prior_source", type=str, default="pretrained", choices=["map", "pretrained"],
                   help="Must match the training run.")
    p.add_argument("--map_suffix", type=str, default=None,
                   help="[prior_source=map] Suffix of the MAP router weights dir used at training.")
    p.add_argument("--num_samples", type=int, default=35,
                   help="MC samples S for stochastic routing (paper: 35). Ignored if --routing deterministic.")
    p.add_argument("--routing", type=str, default="stochastic", choices=["stochastic", "deterministic"],
                   help="stochastic = paper inference (S-sample softmax-averaged routing; PRIMARY). "
                        "deterministic = posterior-mean routing (ABLATION: changes downstream hidden states).")
    # --- what text to analyse ---
    p.add_argument("--split", type=str, default="val", choices=["val", "test"],
                   help="Which MedExQA split to read (source=medexqa). PROTOCOL: 'val' (50 ex) is the "
                        "selection set (sign/threshold/calibrator/NLI audit); 'test' (175 ex) is evaluated "
                        "ONCE per frozen configuration -- pass --split test explicitly and only then.")
    p.add_argument("--source", type=str, default="medexqa",
                   choices=["medexqa", "builtin", "obqa", "textfile"],
                   help="medexqa=generation test set (answer region); builtin=free prose; "
                        "obqa=MCQA prompts; textfile=one passage/line.")
    p.add_argument("--generate", action="store_true",
                   help="[medexqa only] Autoregressively generate the explanation and read the signal "
                        "over the MODEL'S own tokens. Default: teacher-force over the GOLD explanation.")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="[--generate] Max explanation tokens to generate per question.")
    p.add_argument("--text_path", type=str, default=None, help="Required if --source textfile.")
    p.add_argument("--num_examples", type=int, default=50,
                   help="Max examples (medexqa/obqa/textfile). builtin uses all 15.")
    p.add_argument("--chat_template", action="store_true",
                   help="Wrap builtin/textfile text in the MCQA chat template. Ignored for medexqa/obqa "
                        "(which always use their own template).")
    # --- analysis knobs ---
    p.add_argument("--spike_pct", type=float, default=0.15, help="Top/bottom fraction defining a 'spike'.")
    p.add_argument("--seq_last_k", type=int, default=10,
                   help="[--generate] Window for the last-k sequence-level aggregates in the abstention readout.")
    p.add_argument("--max_html_examples", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="results/token_analysis")
    p.add_argument("--tag", type=str, default=None, help="Extra tag appended to output filenames.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Token categorisation (heuristic -- no spaCy dependency).
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


@torch.no_grad()
def records_from_ids(model, tokenizer, input_ids, fcvr_layers, causal_model, device, answer_start=0):
    """One forward pass over a single sequence given its token ids (batch_size=1).

    Returns a list of per-position dicts aligned so index t describes 'position t,
    about to predict token t+1'. When answer_start>0, only positions t whose
    predicted token (t+1) lies in the answer region (t+1 >= answer_start) are
    returned -- i.e. the generated / gold explanation tokens.
    """
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[0]
    if seq_len < 3:
        return []

    logits = model(input_ids=input_ids.unsqueeze(0)).logits[0].float()  # [seq, vocab]

    logp = F.log_softmax(logits, dim=-1)
    probs = logp.exp()
    entropy = -(probs * logp).sum(dim=-1)  # [seq]
    next_ids = input_ids[1:]
    surprisal = -logp[torch.arange(seq_len - 1, device=device), next_ids]  # [seq-1]

    per_layer = []
    for l in fcvr_layers:
        L = causal_model.layers[l].block_sparse_moe.router.last_cholesky_factor
        E = L.shape[-1]
        L = L.view(1, seq_len, E, E)[0].float()          # [seq, E, E]
        per_layer.append((L ** 2).sum(dim=(-1, -2)))      # [seq] = ||L||_F^2 = tr(LL^T)
    if per_layer:
        ilv_layers = torch.stack(per_layer, dim=0)        # [n_layers, seq]
        ilv = ilv_layers.mean(dim=0)                      # [seq]
    else:                                                 # arm=det: no FCVR layers -> no ILV
        ilv_layers = torch.full((0, seq_len), float("nan"), device=device)
        ilv = torch.full((seq_len,), float("nan"), device=device)

    tok_strs = [tokenizer.decode([i]) for i in input_ids.tolist()]

    records = []
    for t in range(seq_len - 1):
        if answer_start and (t + 1) < answer_start:   # keep only answer-region predictions
            continue
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


def collect_per_token(model, tokenizer, text_string, fcvr_layers, causal_model, device):
    """String entry point (builtin/obqa/textfile): tokenize then read every position."""
    ids = tokenizer(text_string, return_tensors="pt", truncation=True, max_length=2048).input_ids[0]
    return records_from_ids(model, tokenizer, ids, fcvr_layers, causal_model, device, answer_start=0)


def _unigram_f1(pred, ref):
    """Whitespace-unigram F1 (dependency-free ROUGE-1 stand-in) between a
    generated explanation and a gold reference."""
    tok = lambda s: [w.strip(string.punctuation).lower() for w in s.split()]
    p = [w for w in tok(pred) if w]
    r = [w for w in tok(ref) if w]
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


class OnlineILVRecorder:
    """Forward hooks on the FCVR routers that capture, at every forward call
    made by `model.generate`, tr(L L^T) per position (the Inf-Logit-Var) and the
    entropy of the actual routing distribution (gate entropy). Call 0 processes
    the whole prompt (N = prompt_len rows), every later call one new token (N=1).
    `per_generated_token()` aligns these to the generated tokens: token t was
    produced by the position that predicted it -- the last prompt row for t=0,
    the single row of call t for t>=1."""

    def __init__(self, causal_model, fcvr_layers):
        self.layers = list(fcvr_layers)
        self.calls = {l: [] for l in self.layers}       # layer -> list of np arrays [N]
        self.gate_ent = {l: [] for l in self.layers}
        self.handles = []
        for l in self.layers:
            router = causal_model.layers[l].block_sparse_moe.router
            self.handles.append(router.register_forward_hook(self._make_hook(l)))

    def _make_hook(self, l):
        def hook(module, inputs, output):
            L = module.last_cholesky_factor.detach().float()            # [N, E, E]
            self.calls[l].append((L ** 2).sum(dim=(-1, -2)).cpu().numpy())
            logits = output[-1].detach().float()                         # [N, E] routing logits actually used
            p = torch.softmax(logits, dim=-1)
            self.gate_ent[l].append((-(p * torch.log(p.clamp(min=1e-12))).sum(-1)).cpu().numpy())
        return hook

    def reset(self):
        for l in self.layers:
            self.calls[l].clear(); self.gate_ent[l].clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def per_generated_token(self, n_generated):
        """-> (ilv [G], ilv_per_layer [n_layers, G], gate_entropy [G]) or None if the
        call pattern does not match (e.g. a cache-less generate that re-runs the
        full sequence each step)."""
        ilv_layers, ge_layers = [], []
        for l in self.layers:
            calls = self.calls[l]
            if not calls or len(calls) != n_generated:
                return None
            if any(c.shape[0] != 1 for c in calls[1:]):
                return None
            ilv_layers.append(np.array([calls[0][-1]] + [c[0] for c in calls[1:]], dtype=float))
            ge = self.gate_ent[l]
            ge_layers.append(np.array([ge[0][-1]] + [c[0] for c in ge[1:]], dtype=float))
        ilv_layers = np.stack(ilv_layers, 0)
        return ilv_layers.mean(0), ilv_layers, np.stack(ge_layers, 0).mean(0)


def _entropy_surprisal_from_scores(scores, gen_ids):
    """Decoding-time predictive entropy / surprisal of each generated token from
    the per-step logits returned by generate(output_logits=True)."""
    ent, sur = [], []
    for step, tok in zip(scores, gen_ids.tolist()):
        logp = F.log_softmax(step[0].float(), dim=-1)
        ent.append(float(-(logp.exp() * logp).sum().item()))
        sur.append(float(-logp[tok].item()))
    return ent, sur


def collect_medexqa(model, tokenizer, args, fcvr_layers, causal_model, device):
    """MedExQA generation source. Teacher-forced over the gold explanation, or
    (with --generate) over the model's own greedily-generated explanation. Only
    the answer (explanation) region is analysed.

    In --generate mode each example additionally gets a sequence-level meta
    record: the generated text, a unigram-F1 quality score against the (up to
    two) gold explanations, and a LETTER-PROBE correctness label -- the model
    is asked the same question in MCQA form and its argmax choice over
    {A,B,C,D} is compared with the dataset's gold answer letter. These labels
    feed the abstention readout (see abstention_analysis)."""
    ds = load_exp_dataset("medexqa", split=args.split)[: args.num_examples]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos = tokenizer.eos_token or ""
    choices = ["A", "B", "C", "D"]
    choice_ids = torch.tensor([tokenizer.convert_tokens_to_ids(c) for c in choices], device=device)

    recorder = OnlineILVRecorder(causal_model, fcvr_layers) if (args.generate and fcvr_layers) else None
    n_online_ok = 0

    per_example, raw_texts, metas = [], [], []
    for ex_i, ex in enumerate(tqdm(ds, desc="generate" if args.generate else "teacher-force")):
        prompt = generation_prompt_engineer(ex, tokenizer=tokenizer)["question"]
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048
        ).input_ids.to(device)
        answer_start = prompt_ids.shape[1]
        online = None
        timing = {}

        # Per-example seed so stochastic routing is reproducible independent of order.
        torch.manual_seed(args.seed * 100003 + ex_i)
        if args.generate:
            if recorder is not None:
                recorder.reset()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen = model.generate(
                prompt_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=pad_id, use_cache=True,
                return_dict_in_generate=True, output_logits=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            timing["gen_seconds"] = time.perf_counter() - t0
            full_ids = gen.sequences[0]                        # prompt + generated
            gen_ids = full_ids[answer_start:]
            n_gen = int(gen_ids.shape[0])
            timing["tokens_per_second"] = n_gen / max(timing["gen_seconds"], 1e-9)
            step_logits = gen.logits if getattr(gen, "logits", None) is not None else gen.scores
            ent_on, sur_on = _entropy_surprisal_from_scores(step_logits, gen_ids)
            if recorder is not None:
                aligned = recorder.per_generated_token(n_gen)
                if aligned is not None:
                    online = {"ilv": aligned[0], "ilv_layers": aligned[1], "gate_ent": aligned[2],
                              "entropy": ent_on, "surprisal": sur_on}
                    n_online_ok += 1
                else:
                    print(f"  (warning: online ILV alignment failed for example {ex_i}; "
                          f"calls={[len(recorder.calls[l]) for l in fcvr_layers][:3]}..., n_gen={n_gen})")
            else:
                online = {"ilv": None, "ilv_layers": None, "gate_ent": None, "entropy": ent_on, "surprisal": sur_on}
        else:
            gold = str(ex["answer"]) + eos
            ans_ids = tokenizer(gold, add_special_tokens=False).input_ids
            full_ids = torch.cat([prompt_ids[0], torch.tensor(ans_ids, device=device)])

        # Post-hoc pass (second teacher-forced forward over the finished sequence).
        torch.manual_seed(args.seed * 100003 + ex_i)
        recs = records_from_ids(
            model, tokenizer, full_ids, fcvr_layers, causal_model, device, answer_start=answer_start
        )
        if args.generate and online is not None and len(recs) == len(online["entropy"]):
            # PRIMARY per-token fields = decoding-time (online); post-hoc kept as *_posthoc.
            for k, r in enumerate(recs):
                r["entropy_posthoc"] = r["entropy"]; r["surprisal_posthoc"] = r["surprisal"]
                r["inf_log_var_posthoc"] = r["inf_log_var"]
                r["entropy"] = online["entropy"][k]; r["surprisal"] = online["surprisal"][k]
                if online["ilv"] is not None:
                    r["inf_log_var"] = float(online["ilv"][k])
                    r["inf_log_var_per_layer"] = [float(v) for v in online["ilv_layers"][:, k].tolist()]
                    r["gate_entropy"] = float(online["gate_ent"][k])
                r["readout"] = "online"
        elif args.generate:
            for r in recs:
                r["readout"] = "posthoc_only"

        meta = {"id": ex.get("id", ""), "gold_letter": ex.get("gold_letter", ""),
                "prompt_len_tokens": int(answer_start)}
        if args.generate:
            gen_ids = full_ids[answer_start:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            eos_id = tokenizer.eos_token_id
            ended_with_eos = bool(eos_id is not None and gen_ids.numel() > 0 and int(gen_ids[-1]) == eos_id)
            meta["n_gen_tokens"] = int(gen_ids.shape[0])
            meta["gen_len_tokens"] = int(gen_ids.shape[0])
            meta["ended_with_eos"] = ended_with_eos
            # Hit the cap without emitting EOS -> the explanation was cut off.
            meta["truncated"] = bool((not ended_with_eos) and gen_ids.shape[0] >= args.max_new_tokens)
            meta["gen_text"] = gen_text  # FULL text (the label script judges it)
            meta.update(timing)
            meta["online_ilv_available"] = bool(online is not None and online.get("ilv") is not None)
            refs = [str(ex["answer"]).strip()] + ([str(ex["explanation_2"]).strip()] if ex.get("explanation_2") else [])
            meta["refs"] = refs
            meta["question"] = ex.get("letter_question") or ex["question"]
            meta["expl_f1"] = max(_unigram_f1(gen_text, r) for r in refs)
            if ex.get("gold_letter") and ex.get("letter_question"):
                probe = multiple_choice_prompt_engineer(
                    {"question": ex["letter_question"], "answer": ex["gold_letter"], "id": meta["id"]},
                    tokenizer=tokenizer,
                )["question"]
                probe_ids = tokenizer(
                    probe, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048
                ).input_ids.to(device)
                with torch.no_grad():
                    letter_logits = model(input_ids=probe_ids).logits[0, -1, :][choice_ids]
                # SECONDARY label: a separate MCQA letter-probe pass (different
                # prompt); it does NOT judge the generated explanation.
                meta["pred_letter_probe"] = choices[int(letter_logits.argmax().item())]
                meta["correct_probe"] = bool(meta["pred_letter_probe"] == ex["gold_letter"])

        per_example.append(recs)
        raw_texts.append(ex["question"][:160])
        metas.append(meta)
    if recorder is not None:
        recorder.remove()
        print(f"Online ILV captured for {n_online_ok}/{len(per_example)} generations.")
    return per_example, raw_texts, metas


def build_prose_texts(args, tokenizer):
    """builtin/obqa/textfile sources -> list of (raw, tokenized_string)."""
    if args.source == "builtin":
        raw = BUILTIN_TEXTS
    elif args.source == "textfile":
        assert args.text_path, "--text_path required for --source textfile"
        with open(args.text_path) as f:
            raw = [ln.strip() for ln in f if ln.strip()][: args.num_examples]
    else:  # obqa
        ds = load_exp_dataset(args.dataset_shortcode, split="test")[: args.num_examples]
        return [(x["question"], multiple_choice_prompt_engineer(x, tokenizer=tokenizer)["question"]) for x in ds]

    out = []
    for t in raw:
        if args.chat_template:
            templated = multiple_choice_prompt_engineer({"question": t, "answer": "", "id": ""}, tokenizer=tokenizer)["question"]
            out.append((t, templated))
        else:
            out.append((t, t))
    return out


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

    corr = {
        "pooled_raw": {
            "spearman_ilv_entropy": _spearman(ilv, ent),
            "spearman_ilv_surprisal": _spearman(ilv, sur),
            "pearson_ilv_entropy": _pearson(ilv, ent),
            "pearson_ilv_surprisal": _pearson(ilv, sur),
            "spearman_entropy_surprisal": _spearman(ent, sur),
        }
    }
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

    # Permutation null for the headline statistic: shuffle ilv WITHIN each
    # example, recompute the per-example-mean Spearman, repeat. This replaces
    # the arbitrary +-0.20 rules of thumb with an empirical reference.
    rng = np.random.default_rng(0)
    pairs = [([r["inf_log_var"] for r in recs], [r["entropy"] for r in recs])
             for recs in per_example_records if len(recs) >= 3]
    null_means = []
    for _ in range(200):
        vals = []
        for ilv_l, ent_l in pairs:
            vals.append(_spearman(rng.permutation(ilv_l), ent_l)[0])
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            null_means.append(float(np.mean(vals)))
    obs = corr["per_example_spearman_ilv_entropy"]["mean"]
    if null_means and not np.isnan(obs):
        null_arr = np.asarray(null_means)
        corr["per_example_spearman_ilv_entropy"]["null_mean"] = float(null_arr.mean())
        corr["per_example_spearman_ilv_entropy"]["null_std"] = float(null_arr.std())
        corr["per_example_spearman_ilv_entropy"]["perm_p_two_sided"] = float(
            np.mean(np.abs(null_arr) >= abs(obs)))

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
            "inf_log_var_z_mean": float(np.mean(d["ilv_z"])),
            "entropy_z_mean": float(np.mean(d["ent_z"])),
        }

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
    # Online vs post-hoc ILV agreement (generate mode): per-example Spearman and
    # mean |diff|. If these are far from 1 / 0, the retrospective readout is NOT a
    # proxy for the decoding-time score.
    ovp = []
    absdiff = []
    for recs in per_example_records:
        if len(recs) >= 3 and all("inf_log_var_posthoc" in r for r in recs):
            a = [r["inf_log_var"] for r in recs]; b = [r["inf_log_var_posthoc"] for r in recs]
            rho = _spearman(a, b)[0]
            if not np.isnan(rho):
                ovp.append(rho)
            absdiff.append(float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))))
    online_vs_posthoc = None
    if ovp:
        online_vs_posthoc = {"n": len(ovp), "per_example_spearman_mean": float(np.mean(ovp)),
                             "per_example_spearman_std": float(np.std(ovp)),
                             "mean_abs_diff": float(np.mean(absdiff))}
    return {"correlation": corr, "category": category, "spikes": spikes, "online_vs_posthoc": online_vs_posthoc}


def verdict(summary):
    r = summary["correlation"]["per_example_spearman_ilv_entropy"]["mean"]
    zr = summary["correlation"]["pooled_zscored"]["spearman_ilv_entropy"][0]
    sp = summary["spikes"]
    lines = []
    lines.append(f"per-example mean Spearman(inf_log_var, entropy) = {r:+.3f}")
    ne = summary["correlation"]["per_example_spearman_ilv_entropy"]
    if "null_mean" in ne:
        lines.append(f"  vs within-example permutation null {ne['null_mean']:+.3f} "
                     f"± {ne['null_std']:.3f}  (two-sided p ≈ {ne['perm_p_two_sided']:.3f})")
    lines.append(f"pooled z-scored Spearman(inf_log_var, entropy)  = {zr:+.3f}")
    ovp = summary.get("online_vs_posthoc")
    if ovp:
        lines.append(f"online vs post-hoc ILV: per-example Spearman {ovp['per_example_spearman_mean']:+.3f} "
                     f"± {ovp['per_example_spearman_std']:.3f}, mean|diff| {ovp['mean_abs_diff']:.4f}  (n={ovp['n']})")
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
               "or treat this as the negative finding.")
    else:
        tag = ("NO TRANSFER (~0) -> the router variance does not track per-token generation "
               "uncertainty even after generation fine-tuning. A valid finding.")
    lines.append("VERDICT: " + tag)
    return "\n".join(lines)


def abstention_analysis(per_example_records, metas, last_k=10):
    """Sequence-level abstention readout (--generate mode only).

    Aggregates the per-token Inf-Logit-Var over each generated explanation
    (mean / max / last / mean & max over the last k tokens) and tests whether
    any aggregate predicts that the model answered the underlying question
    WRONG (letter-probe label). Every ILV aggregate is benchmarked against the
    signals a decoder gets for free: mean/max predictive entropy and
    per-token NLL of the generated sequence. AUROC < 0.5 means the INVERTED
    score is the informative direction (flipped AUROC = 1 - AUROC)."""
    rows = []
    for recs, m in zip(per_example_records, metas):
        if not recs or m.get("correct_probe") is None:
            continue
        online = all(r.get("readout") == "online" for r in recs)
        ilv = [r["inf_log_var"] for r in recs]
        ent = [r["entropy"] for r in recs]
        sur = [r["surprisal"] for r in recs]
        k = min(last_k, len(ilv))

        def _aggs(prefix, v):
            return {f"{prefix}_mean": float(np.mean(v)), f"{prefix}_max": float(np.max(v)),
                    f"{prefix}_last": float(v[-1]), f"{prefix}_mean_last{last_k}": float(np.mean(v[-k:])),
                    f"{prefix}_max_last{last_k}": float(np.max(v[-k:]))}
        scores = {}
        has_ilv = not np.all(np.isnan(ilv))
        # PRIMARY: decoding-time ILV (online) when captured; else the post-hoc series only.
        if has_ilv:
            scores.update(_aggs("ilv_online" if online else "ilv_posthoc", ilv))
            if online and all("inf_log_var_posthoc" in r for r in recs):
                scores.update(_aggs("ilv_posthoc", [r["inf_log_var_posthoc"] for r in recs]))
        scores.update({
            "entropy_mean_BASELINE": float(np.mean(ent)),
            "entropy_max_BASELINE": float(np.max(ent)),
            "nll_per_token_BASELINE": float(np.mean(sur)),
            "gen_len_BASELINE": float(len(recs)),
        })
        if all("gate_entropy" in r for r in recs):
            ge = [r["gate_entropy"] for r in recs]
            scores["gate_entropy_mean_BASELINE"] = float(np.mean(ge))
            scores["gate_entropy_max_BASELINE"] = float(np.max(ge))
        rows.append({
            "readout": "online" if online else "posthoc_only",
            "id": m.get("id", ""),
            # Probe-derived label (secondary). label_generation_correctness.py adds
            # option_correct / NLI / unjudgeable fields to this row.
            "wrong_probe": 0 if m["correct_probe"] else 1,
            "correct_probe": bool(m["correct_probe"]),
            "pred_letter_probe": m.get("pred_letter_probe"),
            "gold_letter": m.get("gold_letter"),
            "expl_f1": m.get("expl_f1"),
            "n_gen_tokens": m.get("n_gen_tokens"),
            "gen_len_tokens": m.get("gen_len_tokens"),
            "prompt_len_tokens": m.get("prompt_len_tokens"),
            "truncated": m.get("truncated"),
            "ended_with_eos": m.get("ended_with_eos"),
            "gen_text": m.get("gen_text"),
            "refs": m.get("refs"),
            "question": m.get("question"),
            "gen_seconds": m.get("gen_seconds"),
            "tokens_per_second": m.get("tokens_per_second"),
            "scores": scores,
        })
    if not rows:
        return None, []

    labels = [r["wrong_probe"] for r in rows]
    f1s = [r["expl_f1"] for r in rows]
    result = {
        "n": len(rows),
        "n_wrong": int(np.sum(labels)),
        "letter_probe_accuracy": float(1.0 - np.mean(labels)),
        "n_truncated": int(sum(1 for r in rows if r.get("truncated"))),
        "label": "wrong_probe (letter-probe; SECONDARY -- run label_generation_correctness.py for option/NLI labels)",
        "auroc_predict_wrong": {},
        "spearman_vs_expl_f1": {},
        "note": ("auroc_predict_wrong: score-high should flag WRONG answers; AUROC < 0.5 "
                 "means the inverted score works (flipped = 1 - AUROC). An ILV aggregate "
                 "must beat the *_BASELINE rows to add value at decode time. "
                 "spearman_vs_expl_f1: negative = high score tracks LOW explanation quality."),
    }
    try:
        from sklearn.metrics import roc_auc_score
        have_sklearn = True
    except Exception:
        have_sklearn = False
    names = sorted({k for r in rows for k in r["scores"]}, key=lambda n: (n.endswith("BASELINE"), n))
    for name in names:
        if any(name not in r["scores"] for r in rows):
            continue  # not available on every row (e.g. online failed for some)
        s = [r["scores"][name] for r in rows]
        if have_sklearn and len(set(labels)) == 2:
            result["auroc_predict_wrong"][name] = float(roc_auc_score(labels, s))
        else:
            result["auroc_predict_wrong"][name] = float("nan")
        result["spearman_vs_expl_f1"][name] = _spearman(s, f1s)[0]
    return result, rows


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _heat(val, vmin, vmax):
    if vmax <= vmin:
        f = 0.0
    else:
        f = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    r, g, b = 255, int(255 * (1 - f)), int(255 * (1 - f))
    return f"rgb({r},{g},{b})"


def write_html(per_example_records, raw_texts, path, max_examples):
    css = ("body{font-family:monospace;background:#111;color:#eee;padding:16px;}"
           ".tok{padding:1px 2px;border-radius:2px;color:#000;}"
           ".lbl{color:#7fd;margin:14px 0 4px;font-weight:bold;}"
           ".cat{outline:2px solid #06f;}")
    parts = [f"<!doctype html><meta charset='utf-8'><style>{css}</style>",
             "<h2>Step 1: per-token signals (top row = predictive ENTROPY, bottom row = INF-LOG-VAR)</h2>",
             "<p>Redder = higher. Blue outline = number/entity/connective. "
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

    if args.arm != "det" and not args.swap_layers:
        raise SystemExit("--swap_layers is required unless --arm det")
    model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=args.kvq_adapter_path, device_map="cuda:0")
    tokenizer = load_tokenizer(args.model_shortcode)
    model, fcvr_layers = prepare_model_by_arm(model, args)  # fcvr: MAP (if map-prior) + trained FCVR weights

    causal_model = model.base_model.model.model
    print(f"Arm: {args.arm}")
    det = args.routing == "deterministic"
    for l in fcvr_layers:
        causal_model.layers[l].block_sparse_moe.router.deterministic_readout = det
    print(f"Routing mode: {'DETERMINISTIC posterior-mean (ABLATION)' if det else f'STOCHASTIC S={args.num_samples} (paper inference; PRIMARY)'}")

    if args.source == "medexqa":
        mode = "generate" if args.generate else "teacher_forced"
        if args.split == "test":
            print("#" * 72 + "\n# TEST SPLIT (175 ex): evaluate ONCE per frozen configuration. Selection\n"
                  "# (sign / thresholds / calibration / label audit) must already be frozen on val.\n" + "#" * 72)
        print(f"MedExQA source | split={args.split} | mode={mode} | answer-region only | {args.num_examples} examples")
        per_example, raw_texts, metas = collect_medexqa(model, tokenizer, args, fcvr_layers, causal_model, model.device)
    else:
        mode = args.source
        texts = build_prose_texts(args, tokenizer)
        print(f"Analysing {len(texts)} examples from source='{args.source}' "
              f"(chat_template={'yes' if (args.chat_template or args.source == 'obqa') else 'no'})")
        per_example, raw_texts, metas = [], [], None
        for raw, tokstr in tqdm(texts, desc="forward"):
            per_example.append(collect_per_token(model, tokenizer, tokstr, fcvr_layers, causal_model, model.device))
            raw_texts.append(raw)

    all_records = [r for recs in per_example for r in recs]
    print(f"Collected {len(all_records)} token positions.")

    summary = analyze(all_records, per_example, args.spike_pct)
    summary["config"] = {
        "arm": args.arm,
        "source": args.source, "split": args.split if args.source == "medexqa" else None,
        "mode": mode, "num_examples": len(per_example),
        "n_tokens": len(all_records), "fcvr_layers": fcvr_layers,
        "prior_source": args.prior_source, "run_suffix": args.run_suffix,
        "routing": "deterministic" if det else f"stochastic_S{args.num_samples}",
        "num_samples": None if det else args.num_samples, "seed": args.seed,
        "primary_readout": "online" if (args.source == "medexqa" and args.generate) else "teacher_forced",
        "max_new_tokens": args.max_new_tokens if (args.source == "medexqa" and args.generate) else None,
    }
    v = verdict(summary)
    summary["verdict"] = v

    # Sequence-level abstention readout (generate mode only): does an
    # aggregate of the per-token signal predict a WRONG letter-probe answer?
    seq_rows = []
    if args.source == "medexqa" and args.generate and metas is not None:
        abst, seq_rows = abstention_analysis(per_example, metas, last_k=args.seq_last_k)
        if abst is not None:
            summary["abstention"] = abst
        secs = [m["gen_seconds"] for m in metas if m.get("gen_seconds") is not None]
        tps = [m["tokens_per_second"] for m in metas if m.get("tokens_per_second") is not None]
        summary["runtime"] = {
            "n": len(secs), "gen_seconds_mean": float(np.mean(secs)) if secs else None,
            "gen_seconds_total": float(np.sum(secs)) if secs else None,
            "tokens_per_second_mean": float(np.mean(tps)) if tps else None,
            "online_ilv_available_frac": float(np.mean([bool(m.get("online_ilv_available")) for m in metas])),
            "n_truncated": int(sum(1 for m in metas if m.get("truncated"))),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    # Historical test-split files carry no split token; val-split outputs are
    # marked so the two can never be confused or overwrite each other.
    split_tag = f"_{args.split}" if (args.source == "medexqa" and args.split != "test") else ""
    base = os.path.join(args.output_dir, f"step1_{args.source}{split_tag}_{mode}{tag}")
    with open(base + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(base + "_pertoken.jsonl", "w") as f:
        for ex_i, recs in enumerate(per_example):
            for r in recs:
                rr = {k: val for k, val in r.items() if k != "inf_log_var_per_layer"}
                rr["example"] = ex_i
                f.write(json.dumps(rr) + "\n")
    if seq_rows:
        with open(base + "_seqlevel.jsonl", "w") as f:
            for r in seq_rows:
                f.write(json.dumps(r) + "\n")
    write_html(per_example, raw_texts, base + ".html", args.max_html_examples)
    write_plots(all_records, summary, base + ".png")

    print("\n" + "=" * 70)
    print(v)
    print("=" * 70)
    print("\nPer-category (z-scored within example; >0 = elevated):")
    for c, d in summary["category"].items():
        print(f"  {c:11s} n={d['n']:5d}  ilv_z={d['inf_log_var_z_mean']:+.3f}  ent_z={d['entropy_z_mean']:+.3f}")
    if summary.get("abstention"):
        a = summary["abstention"]
        print(f"\nSequence-level abstention readout "
              f"(n={a['n']}, wrong={a['n_wrong']}, letter-probe acc={a['letter_probe_accuracy']:.3f}):")
        for name, auc in a["auroc_predict_wrong"].items():
            flip = (1.0 - auc) if not np.isnan(auc) else float("nan")
            print(f"  AUROC(wrong)[{name:26s}] = {auc:.3f}   flipped = {flip:.3f}   "
                  f"spearman_vs_f1 = {a['spearman_vs_expl_f1'][name]:+.3f}")
        print("  (ILV aggregates must beat the *_BASELINE rows to matter.)")
    print(f"\nSaved: {base}.json / _pertoken.jsonl / .html / .png"
          + (" / _seqlevel.jsonl" if seq_rows else ""))


if __name__ == "__main__":
    main()

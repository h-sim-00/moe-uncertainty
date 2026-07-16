"""
VTSR (Variational Temperature Scaling Router) evaluation -- faithful reconstruction.

Mirrors how vtsr-tuning.py trains: loads the fine-tuned MAP routers into ALL 32
layers, then swaps the trained (susceptible) layers to VTSR and loads their
temperature-net weights. Because the MAP routers are loaded first, each VTSR
layer's frozen deterministic logits `l_det = self.layer(u)` come from the same
fine-tuned MAP router used during training, and every non-VTSR layer stays MAP.

Three OoD signals are extracted (paper Table 2, VTSR row):
  - answer_entropy : Shannon entropy of the predictive softmax over {A,B,C,D}
                     at the final token (plumbing sanity signal).
  - gate_ent       : H(softmax(l_det / T_phi)) -- entropy of the TEMPERATURE-SCALED
                     routing distribution over the N experts, at the final token,
                     averaged over the VTSR layers. (Paper Eq. 29 for VTSR; the
                     strong VTSR signal, avg AUROC ~0.743.)
  - inf_temp       : the raw learned scalar temperature T_phi (paper Eq. 31),
                     at the final token, averaged over the VTSR layers. Higher T
                     means the router wants a flatter distribution. (The VTSR-
                     specific signal; deliberately weak in the paper, ~0.509.)

This file is dedicated to VTSR and leaves the generic evaluate.py untouched.
"""

import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import setup_environment, load_exp_dataset, multiple_choice_prompt_engineer
from utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll
from model import load_peft_model_and_adapter, load_tokenizer
from model.adapters import granite_adapter

# Paper Table 2 OoD targets (OBQA is the fixed ID anchor).
OOD_DATASETS = {
    "arc_e": "near",
    "arc_c": "near",
    "medmcqa_med": "far",
    "mmlu_law": "far",
}

# One-time padding sanity check (see also standalone debug_padding.py).
_PADDING_REPORTED = False


def report_padding(tokenizer, inputs):
    """Print ONCE whether the `[:, -1, :]` final-token readout used below grabs the
    real last token for every row, or a PAD token (right-padding). Works for any
    padding side: the real last-token index is the last position whose mask == 1."""
    global _PADDING_REPORTED
    if _PADDING_REPORTED:
        return
    _PADDING_REPORTED = True

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    bsz, seq_len = ids.shape
    idx = torch.arange(seq_len, device=mask.device)
    last_real = (mask * idx).argmax(dim=1)      # last index with mask==1, per row
    minus1_idx = seq_len - 1                     # what `[:, -1]` grabs
    n_ok = int((last_real == minus1_idx).sum().item())

    print("\n" + "=" * 66)
    print("[PADDING CHECK] (printed once per eval run)")
    print(f"  tokenizer.padding_side = {tokenizer.padding_side!r}")
    print(f"  pad_token = {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"  batch shape (rows x seq_len) = {bsz} x {seq_len}")
    print(f"  real last-token index per row = {last_real.tolist()}")
    print(f"  rows where `-1` == real last token: {n_ok}/{bsz}")
    for r in range(min(3, bsz)):
        lr = last_real[r].item()
        flag = "OK" if lr == minus1_idx else "MISMATCH"
        print(f"    row {r}: [-1] -> {tokenizer.decode(ids[r, -1:].tolist())!r}   "
              f"real-last(idx {lr}) -> {tokenizer.decode(ids[r, lr:lr + 1].tolist())!r}   [{flag}]")
    if n_ok == bsz:
        print("  VERDICT: `-1` is the real final token for EVERY row -> extraction CORRECT.")
    else:
        print("  VERDICT: `-1` hits a PAD token on some rows -> extraction WRONG (right-padded).")
        print("           Signals for those rows are read at a pad position; use a mask-based readout.")
    print("=" * 66 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Faithful VTSR evaluation (answer_entropy + gate_ent + inf_temp).")
    parser.add_argument("--task", type=str, required=True, choices=["id_calibration", "ood_detection"])
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--dataset_shortcode", type=str, required=True,
                        help="The ID dataset the model was trained on (e.g. 'obqa').")
    parser.add_argument("--kvq_adapter_path", type=str, required=True,
                        help="Path to the Stage-1 fine-tuned KVQ LoRA adapter.")
    parser.add_argument("--swap_layers", type=int, nargs="+", required=True,
                        help="Layers that carry a trained VTSR router.")
    parser.add_argument("--temperature_mode", type=str, default="shared",
                        help="Must match the training run: names the router_weights/vtsr_<mode>/ dir.")
    parser.add_argument("--run_suffix", type=str, default="",
                        help="Suffix on the VTSR weights dir; must match the training run's --run_suffix.")
    parser.add_argument("--output_json_path", type=str, required=True)

    parser.add_argument("--deterministic_routing", action="store_true",
                        help="DIAGNOSTIC: use deterministic Top-K at inference instead of the "
                             "paper's stochastic Sample-K. If ID accuracy recovers with this on, "
                             "the collapse is inflated-T + random sampling (not a wiring bug).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_model_vtsr(model, args):
    """Reconstruct the trained model exactly as vtsr-tuning.py set it up:
    all 32 layers -> fine-tuned MAP routers (also seeds each VTSR layer's frozen
    deterministic logits), then the selected layers -> VTSR with trained weights."""
    print("--- Preparing VTSR model (MAP-prior, matching training) ---")
    # 1. Fine-tuned MAP routers on all 32 layers.
    model = granite_adapter.load_granite_map_routers(model, args=args)
    # 2. Swap the selected layers to VTSR and load their trained temperature nets.
    #    load_granite_bayesian_routers reads swap_layers + temperature_mode +
    #    run_suffix and loads
    #    ./router_weights/vtsr_<mode>/vtsr-<model>-<dataset>-<suffix>/layer_<i>_weights.pt
    model = granite_adapter.load_granite_bayesian_routers(model, method="vtsr", args=args)

    # Optional diagnostic: force deterministic Top-K at inference on every VTSR layer.
    if getattr(args, "deterministic_routing", False):
        causal_model = model.base_model.model.model
        for l in args.swap_layers:
            causal_model.layers[l].block_sparse_moe.router.deterministic_inference = True
        print("[DIAGNOSTIC] deterministic_routing=ON -> VTSR inference uses Top-K, not Sample-K.")

    model.eval()
    print(f"VTSR layers: {sorted(args.swap_layers)}")
    return model


def compute_signals(model, tokenizer, dataset, vtsr_layers, args):
    """Single forward pass per batch -> (answer_entropy, gate_ent, inf_temp) np arrays."""
    model.eval()
    causal_model = model.base_model.model.model

    choices = ["A", "B", "C", "D"]
    choice_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(c) for c in choices], device=model.device
    )

    processed = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in dataset]
    questions = [x["question"] for x in processed]

    ans_entropy, gate_ent, inf_temp = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), args.batch_size), desc="Signals"):
            batch_q = questions[i:i + args.batch_size]
            inputs = tokenizer(
                batch_q, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(model.device)
            bsz, seq_len = inputs["input_ids"].shape
            report_padding(tokenizer, inputs)  # one-time diagnostic on the first batch

            logits = model(**inputs).logits

            # --- answer entropy over {A,B,C,D} at the final token ---
            choice_logits = logits[:, -1, :][:, choice_ids]
            probs = F.softmax(choice_logits, dim=1)
            ent = torch.distributions.Categorical(probs=probs).entropy()
            ans_entropy.append(ent.cpu())

            # --- gate_ent + inf_temp at the final token, averaged over VTSR layers ---
            per_layer_ent, per_layer_temp = [], []
            for l in vtsr_layers:
                router = causal_model.layers[l].block_sparse_moe.router
                # last_scaled_logits: [bsz*seq, N] = l_det / T_phi ; last_temperature: [bsz*seq, 1]
                sl = router.last_scaled_logits.view(bsz, seq_len, -1)[:, -1, :]   # [bsz, N]
                t = router.last_temperature.view(bsz, seq_len, -1)[:, -1, :]      # [bsz, 1]
                gate_probs = F.softmax(sl.float(), dim=-1)
                per_layer_ent.append(torch.distributions.Categorical(probs=gate_probs).entropy())  # [bsz]
                per_layer_temp.append(t.squeeze(-1).float())                                        # [bsz]
            gate_ent.append(torch.stack(per_layer_ent, dim=0).mean(dim=0).cpu())
            inf_temp.append(torch.stack(per_layer_temp, dim=0).mean(dim=0).cpu())

    return (torch.cat(ans_entropy).numpy(),
            torch.cat(gate_ent).numpy(),
            torch.cat(inf_temp).numpy())


def run_id_calibration(model, tokenizer, args):
    print("\n--- Task: ID Calibration ---")
    test_dataset = load_exp_dataset(args.dataset_shortcode, split="test")
    _, probs, labels = get_model_predictions(model, tokenizer, test_dataset, batch_size=args.batch_size)

    preds = torch.argmax(probs, dim=1)
    acc = calculate_accuracy(preds, labels).item()
    nll = calculate_nll(probs, labels).item()
    ece, mce = calculate_ece_mce(probs, labels)
    result = {"ACC": acc, "NLL": nll, "ECE": ece.item(), "MCE": mce.item()}
    print(f"{args.dataset_shortcode}: {result}")
    return {args.dataset_shortcode: result}


def _auc(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def run_ood_detection(model, tokenizer, args):
    print("\n--- Task: OOD Detection (answer_entropy + gate_ent + inf_temp) ---")
    vtsr_layers = sorted(args.swap_layers)

    print(f"ID anchor: {args.dataset_shortcode}")
    id_dataset = load_exp_dataset(args.dataset_shortcode, split="test")
    id_ae, id_ge, id_it = compute_signals(model, tokenizer, id_dataset, vtsr_layers, args)
    print(f"  ID mean  answer_entropy={id_ae.mean():.4f}  gate_ent={id_ge.mean():.4f}  inf_temp={id_it.mean():.4f}")

    results = {}
    for ood_code, shift_type in OOD_DATASETS.items():
        print(f"OOD vs {ood_code} ({shift_type})")
        ood_dataset = load_exp_dataset(ood_code, split="test")
        ood_ae, ood_ge, ood_it = compute_signals(model, tokenizer, ood_dataset, vtsr_layers, args)

        ae_auroc, ae_auprc = _auc(id_ae, ood_ae)
        ge_auroc, ge_auprc = _auc(id_ge, ood_ge)
        it_auroc, it_auprc = _auc(id_it, ood_it)
        results[ood_code] = {
            "shift_type": shift_type,
            "answer_entropy": {"auroc": ae_auroc, "auprc": ae_auprc},
            "gate_ent": {"auroc": ge_auroc, "auprc": ge_auprc},
            "inf_temp": {"auroc": it_auroc, "auprc": it_auprc},
        }
        for name, id_s, ood_s, auroc, auprc in [
            ("answer_entropy", id_ae, ood_ae, ae_auroc, ae_auprc),
            ("gate_ent", id_ge, ood_ge, ge_auroc, ge_auprc),
            ("inf_temp", id_it, ood_it, it_auroc, it_auprc),
        ]:
            direction = "OoD>ID" if ood_s.mean() > id_s.mean() else "OoD<ID INVERTED"
            print(f"  {name:<15} AUROC={auroc:.4f} AUPRC={auprc:.4f} (OoD mean {ood_s.mean():.4f}, {direction})")
    return results


def main():
    setup_environment()
    args = parse_args()
    torch.manual_seed(args.seed)

    model = load_peft_model_and_adapter(
        args.model_shortcode, adapter_path=args.kvq_adapter_path, device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    model = prepare_model_vtsr(model, args)

    if args.task == "id_calibration":
        final_results = run_id_calibration(model, tokenizer, args)
    else:
        final_results = run_ood_detection(model, tokenizer, args)

    os.makedirs(os.path.dirname(args.output_json_path) or ".", exist_ok=True)
    with open(args.output_json_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\nSaved results to {args.output_json_path}")


if __name__ == "__main__":
    main()

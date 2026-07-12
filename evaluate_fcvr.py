"""
FCVR (VGLR-FC) evaluation -- faithful reconstruction.

Loads the fine-tuned MAP routers into ALL 32 layers, then swaps the selected
layers to FCVR and loads their trained variational weights. Because the MAP
routers are loaded first, each FCVR layer's frozen `mean_base` is seeded from
the fine-tuned MAP router (matching how fcvr-tuning.py trained), and every
non-FCVR layer stays as fine-tuned MAP.

Two OoD signals are extracted:
  - answer_entropy : Shannon entropy of the predictive softmax over {A,B,C,D}
                     at the final token.
  - inf_log_var    : tr(posterior cov) = ||L||_F^2 of the FCVR router's Cholesky
                     factor at the final token, averaged over the FCVR layers.
                     (The FCVR-specific "Inf-Logit-Var" signal.)

This file is dedicated to FCVR and leaves the generic evaluate.py untouched.
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

# Paper Table 8 OoD targets (OBQA is the fixed ID anchor). Edit if needed.
OOD_DATASETS = {
    "arc_c": "near",
    "arc_e": "near",
    "medmcqa_med": "far",
    "mmlu_law": "far",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Faithful FCVR evaluation (answer_entropy + inf_log_var).")
    parser.add_argument("--task", type=str, required=True, choices=["id_calibration", "ood_detection"])
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--dataset_shortcode", type=str, required=True,
                        help="The ID dataset the model was trained on (e.g. 'obqa').")
    parser.add_argument("--kvq_adapter_path", type=str, required=True,
                        help="Path to the Stage-1 fine-tuned KVQ LoRA adapter.")
    parser.add_argument("--swap_layers", type=int, nargs="+", required=True,
                        help="Layers that carry a trained FCVR router.")
    parser.add_argument("--run_suffix", type=str, default=None,
                        help="Suffix on the FCVR weights dir; must match the training run's --run_suffix.")
    parser.add_argument("--output_json_path", type=str, required=True)

    parser.add_argument("--num_samples", type=int, default=35, help="MC samples for FCVR inference.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_model_fcvr(model, args):
    """Faithful reconstruction: fine-tuned MAP on all layers, FCVR on swap_layers."""
    print("--- Preparing FCVR model (faithful MAP reconstruction) ---")
    # 1. Fine-tuned MAP routers into all 32 layers (also seeds FCVR mean_base).
    model = granite_adapter.load_granite_map_routers(model, args=args)
    # 2. Swap the selected layers to FCVR and load their trained weights.
    #    load_granite_bayesian_routers reads swap_layers and loads
    #    ./router_weights/fcvr/fcvr-<model>-<dataset>/layer_<i>_weights.pt
    model = granite_adapter.load_granite_bayesian_routers(model, method="fcvr", args=args)

    # 3. Set the number of MC samples used at inference on each FCVR router.
    causal_model = model.base_model.model.model
    for i in args.swap_layers:
        router = causal_model.layers[i].block_sparse_moe.router
        if hasattr(router, "num_mc_samples_inference"):
            router.num_mc_samples_inference = args.num_samples

    model.eval()
    print(f"FCVR layers: {sorted(args.swap_layers)} | MC samples: {args.num_samples}")
    return model


def compute_signals(model, tokenizer, dataset, fcvr_layers, args):
    """Single forward pass per batch -> (answer_entropy, inf_log_var) as np arrays."""
    model.eval()
    causal_model = model.base_model.model.model

    choices = ["A", "B", "C", "D"]
    choice_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(c) for c in choices], device=model.device
    )

    processed = [multiple_choice_prompt_engineer(x, tokenizer=tokenizer) for x in dataset]
    questions = [x["question"] for x in processed]

    ans_entropy, inf_log_var = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), args.batch_size), desc="Signals"):
            batch_q = questions[i:i + args.batch_size]
            inputs = tokenizer(
                batch_q, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(model.device)
            bsz, seq_len = inputs["input_ids"].shape

            logits = model(**inputs).logits

            # --- answer entropy over {A,B,C,D} at the final token ---
            choice_logits = logits[:, -1, :][:, choice_ids]
            probs = F.softmax(choice_logits, dim=1)
            ent = torch.distributions.Categorical(probs=probs).entropy()
            ans_entropy.append(ent.cpu())

            # --- inf-log-var: ||L||_F^2 at the final token, averaged over FCVR layers ---
            per_layer = []
            for l in fcvr_layers:
                router = causal_model.layers[l].block_sparse_moe.router
                L = router.last_cholesky_factor  # [bsz*seq, E, E], (batch, seq) row-major
                E = L.shape[-1]
                L_last = L.view(bsz, seq_len, E, E)[:, -1, :, :]  # final token per sequence
                per_layer.append((L_last ** 2).sum(dim=(-1, -2)))  # trace(LL^T) = ||L||_F^2
            inf_log_var.append(torch.stack(per_layer, dim=0).mean(dim=0).cpu())

    return torch.cat(ans_entropy).numpy(), torch.cat(inf_log_var).numpy()


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
    print("\n--- Task: OOD Detection (answer_entropy + inf_log_var) ---")
    fcvr_layers = sorted(args.swap_layers)

    print(f"ID anchor: {args.dataset_shortcode}")
    id_dataset = load_exp_dataset(args.dataset_shortcode, split="test")
    id_ent, id_ilv = compute_signals(model, tokenizer, id_dataset, fcvr_layers, args)
    print(f"  ID mean  answer_entropy={id_ent.mean():.4f}  inf_log_var={id_ilv.mean():.4f}")

    results = {}
    for ood_code, shift_type in OOD_DATASETS.items():
        print(f"OOD vs {ood_code} ({shift_type})")
        ood_dataset = load_exp_dataset(ood_code, split="test")
        ood_ent, ood_ilv = compute_signals(model, tokenizer, ood_dataset, fcvr_layers, args)

        ae_auroc, ae_auprc = _auc(id_ent, ood_ent)
        iv_auroc, iv_auprc = _auc(id_ilv, ood_ilv)
        results[ood_code] = {
            "shift_type": shift_type,
            "answer_entropy": {"auroc": ae_auroc, "auprc": ae_auprc},
            "inf_log_var": {"auroc": iv_auroc, "auprc": iv_auprc},
        }
        print(f"  answer_entropy AUROC={ae_auroc:.4f} AUPRC={ae_auprc:.4f} "
              f"(OoD mean {ood_ent.mean():.4f}, {'OoD>ID' if ood_ent.mean() > id_ent.mean() else 'OoD<ID INVERTED'})")
        print(f"  inf_log_var    AUROC={iv_auroc:.4f} AUPRC={iv_auprc:.4f} "
              f"(OoD mean {ood_ilv.mean():.4f}, {'OoD>ID' if ood_ilv.mean() > id_ilv.mean() else 'OoD<ID INVERTED'})")
    return results


def main():
    setup_environment()
    args = parse_args()
    torch.manual_seed(args.seed)

    model = load_peft_model_and_adapter(
        args.model_shortcode, adapter_path=args.kvq_adapter_path, device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    model = prepare_model_fcvr(model, args)

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

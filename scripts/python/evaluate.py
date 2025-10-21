import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# --- Assumed Utility and Model Imports ---
from utils import setup_environment, load_exp_dataset
from model import load_peft_model_and_adapter, load_tokenizer
from utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll

# --- Import All Router Classes ---
from model.routers.base import MoERouter
from model.routers.mcdr import MCDropoutRouter
from model.routers.mfvr import MeanFieldVariationalRouter
from model.routers.fcvr import FullCovarianceVariationalRouter
from model.routers.vtsr import VariationalTemperatureRouter

def parse_args():
    """Parses command-line arguments for the unified evaluation script."""
    parser = argparse.ArgumentParser(description="Unified evaluation script for all MoE router methods.")
    
    # --- Core Arguments ---
    parser.add_argument("--method", type=str, required=True, 
                        choices=['zero_shot', 'kvq_ft', 'det', 'temp_sampling', 'mcdr', 
                                 'mfvr', 'fcvr', 'vtsr'])
    parser.add_argument("--task", type=str, required=True, choices=['id_calibration', 'ood_detection'])
    parser.add_argument("--model_shortcode", type=str, required=True)
    parser.add_argument("--dataset_shortcode", type=str, required=True, help="The primary ID dataset the model was trained on (e.g., 'obqa').")
    parser.add_argument("--output_json_path", type=str, required=True, help="Path to save the final results JSON file.")
    
    # --- Path Arguments ---
    parser.add_argument("--kvq_adapter_path", type=str, help="Path to the Stage 1 fine-tuned KVQ LoRA adapter.")
    parser.add_argument("--router_weights_path", type=str, help="Path to the directory or file of saved Bayesian router weights/objects.")

    # --- Method-Specific Hyperparameters ---
    parser.add_argument("--swap_layers", type=int, nargs='*', default=None, help="Specific layers to swap with a Bayesian router. Defaults to all.")
    # (4) num_samples default changed to 35
    parser.add_argument("--num_samples", type=int, default=35, help="Number of MC samples for stochastic methods.")
    # (1) dropout_rate default changed to 0.05
    parser.add_argument("--dropout_rate", type=float, default=0.05, help="Dropout rate for re-instantiating MCDR.")
    # (3) temperature default changed to 0.3
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for temperature sampling.")
    parser.add_argument("--vtsr_mode", type=str, choices=['per_expert', 'shared'], help="Mode for VTSR.")

    # --- General ---
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def prepare_model(model, args):
    """
    Dispatcher function to perform the correct 'Lego swap' and load weights
    based on the specified method and swap_layers.
    """
    print(f"--- Preparing model for method: {args.method} ---")
    causal_model = model.base_model.model.model
    device = model.device
    num_layers = len(causal_model.layers)
    swap_layers = args.swap_layers if args.swap_layers is not None else range(num_layers)
    print(f"Targeting layers for swap: {list(swap_layers)}")

    # Define a mapping from method string to router class
    router_class_map = {
        'det': MoERouter,
        'temp_sampling': MoERouter,
        'mcdr': MCDropoutRouter,
        # 'der': MoERouter, # (2) Commented out
        # 'swagr': SwagRouter, # (2) Commented out
        'mfvr': MeanFieldVariationalRouter,
        'fcvr': FullCovarianceVariationalRouter,
        'vtsr': VariationalTemperatureRouter,
    }

    if args.method not in router_class_map:
        return model # No swap needed for zero_shot or kvq_ft

    # Perform the swap for the specified layers
    for i in swap_layers:
        layer = causal_model.layers[i]
        RouterClass = router_class_map[args.method]
        
        init_kwargs = {'config': causal_model.config, 'existing_router': layer.block_sparse_moe.router}
        if args.method == 'temp_sampling':
            init_kwargs['mode'] = 'sample_k'
            init_kwargs['temp'] = args.temperature
        elif args.method == 'mcdr':
            init_kwargs['dropout_rate'] = args.dropout_rate
        elif args.method == 'vtsr':
            init_kwargs['temperature_mode'] = args.vtsr_mode

        new_router = RouterClass(**init_kwargs).to(device)
        
        if args.router_weights_path:
            weights_path = os.path.join(args.router_weights_path, f"layer_{i}_weights.pt")
            if os.path.exists(weights_path):
                new_router.load_weights(weights_path, device=device)
            else:
                print(f"Warning: No weights file found for layer {i} at {weights_path}")
        
        layer.block_sparse_moe.router = new_router

    model.eval()
    print("Model preparation complete.")
    return model

def get_predictions(model, tokenizer, dataset, args):
    """
    Dispatcher for inference. Calls the correct evaluation logic based on the method.
    """
    # --- Configure Routers Before Inference ---
    # For all methods with internal sampling, set the number of samples.
    if args.method in ['mcdr', 'mfvr', 'fcvr']:
        for layer in model.base_model.model.model.layers:
            router = layer.block_sparse_moe.router
            if hasattr(router, 'num_mc_samples_inference'):
                router.num_mc_samples_inference = args.num_samples
            # For MCDR, we still need to ensure dropout is active
            if args.method == 'mcdr' and hasattr(router, 'dropout'):
                router.dropout.train()

    # --- Perform Inference ---
    # For all other methods (including MCDR, MFVR, FCVR), a single forward pass is now sufficient
    # because the sampling logic is encapsulated within the router's eval() forward pass.
    _, probs, labels = get_model_predictions(model, tokenizer, dataset, batch_size=args.batch_size)
    
    return probs, labels

def run_id_calibration(model, tokenizer, args):
    """Runs the In-Distribution Calibration task."""
    print("\n--- Running Task: ID Calibration ---")
    results = {}
    id_datasets = [args.dataset_shortcode]
    for dataset_code in id_datasets:
        print(f"Evaluating on: {dataset_code}")
        test_dataset = load_exp_dataset(dataset_code, split="test")
        probs, labels = get_predictions(model, tokenizer, test_dataset, args)
        
        preds = torch.argmax(probs, dim=1)
        acc = calculate_accuracy(preds, labels).item()
        nll = calculate_nll(probs, labels).item()
        ece, mce = calculate_ece_mce(probs, labels)
        
        results[dataset_code] = {'ACC': acc, 'NLL': nll, 'ECE': ece.item(), 'MCE': mce.item()}
        print(results[dataset_code])
    return results

def run_ood_detection(model, tokenizer, args):
    """Runs the Out-of-Distribution Detection task."""
    print("\n--- Running Task: OOD Detection ---")
    results = {}
    
    id_dataset = load_exp_dataset("obqa", split="test")
    id_probs, _ = get_predictions(model, tokenizer, id_dataset, args)
    id_entropy = torch.distributions.Categorical(probs=id_probs).entropy().numpy()
    
    ood_datasets = {"arc_c": "small", "medmcqa_med": "large", "mmlu_law": "large", "sciq": "large"}
    for ood_code, shift_type in ood_datasets.items():
        print(f"Evaluating OOD against: {ood_code} ({shift_type} shift)")
        ood_dataset = load_exp_dataset(ood_code, split="test")
        ood_probs, _ = get_predictions(model, tokenizer, ood_dataset, args)
        ood_entropy = torch.distributions.Categorical(probs=ood_probs).entropy().numpy()
        
        scores = np.concatenate([id_entropy, ood_entropy])
        labels = np.concatenate([np.zeros_like(id_entropy), np.ones_like(ood_entropy)])
        
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        
        results[ood_code] = {'auroc': auroc, 'auprc': auprc, 'shift_type': shift_type}
        print(results[ood_code])
    return results

def main():
    """Main orchestration script."""
    setup_environment()
    args = parse_args()

    if args.method in ['zero_shot']:
        model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=None, device_map="cuda:0")
    else:
        model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=args.kvq_adapter_path, device_map="cuda:0")
    
    tokenizer = load_tokenizer(args.model_shortcode)

    model = prepare_model(model, args)

    if args.task == 'id_calibration':
        final_results = run_id_calibration(model, tokenizer, args)
    elif args.task == 'ood_detection':
        final_results = run_ood_detection(model, tokenizer, args)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    with open(args.output_json_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nSaved final results to {args.output_json_path}")

if __name__ == "__main__":
    main()
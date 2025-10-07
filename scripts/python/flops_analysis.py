import argparse
import os
import torch
import json
from fvcore.nn import FlopCountAnalysis

# --- Assumed Utility and Model Imports ---
from utils import setup_environment
from model import load_peft_model_and_adapter, load_tokenizer

# --- Import All Router Classes ---
from model.routers.base import MoERouter
from model.routers.mcdr import MCDropoutRouter
from model.routers.mfvr import MeanFieldVariationalRouter
from model.routers.fcvr import FullCovarianceVariationalRouter
from model.routers.vtsr import VariationalTemperatureRouter

def parse_args():
    """Parses command-line arguments for the FLOPs analysis script."""
    parser = argparse.ArgumentParser(description="FLOPs analysis for MoE router methods.")
    parser.add_argument("--model_shortcode", type=str, required=True)
    parser.add_argument("--kvq_adapter_path", type=str, required=True)
    parser.add_argument("--map_router_path", type=str, required=True, help="Path to the directory of saved MAP router weights for initialization.")
    parser.add_argument("--output_json_path", type=str, required=True)
    
    # --- Hyperparameters needed to instantiate models correctly ---
    parser.add_argument("--num_samples", type=int, default=35)
    parser.add_argument("--dropout_rate", type=float, default=0.05)
    parser.add_argument("--vtsr_mode", type=str, default="per_expert", choices=['per_expert', 'shared'])
    
    return parser.parse_args()

def prepare_model(model, method, map_router_path, args):
    """
    Performs the 'Lego swap' to configure the model with the correct router,
    initialized from the provided MAP weights.
    """
    causal_model = model.base_model.model.model
    device = model.device
    
    router_class_map = {
        'det': MoERouter, 'mcdr': MCDropoutRouter, 'mfvr': MeanFieldVariationalRouter,
        'fcvr': FullCovarianceVariationalRouter, 'vtsr': VariationalTemperatureRouter,
    }

    if method not in router_class_map:
        raise ValueError(f"Unknown method for FLOPs analysis: {method}")

    for i, layer in enumerate(causal_model.layers):
        if i <= 21:
            continue
        # First, create a temporary MAP router and load its weights
        map_router = MoERouter(config=causal_model.config)
        weights_path = os.path.join(map_router_path, f"layer_{i}_weights.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"MAP weights not found for layer {i} at {weights_path}.")
        map_router.load_weights(weights_path, device=device)

        # Now, instantiate the correct router class, initializing it from the MAP router
        RouterClass = router_class_map[method]
        init_kwargs = {'config': causal_model.config, 'existing_router': map_router}
        if method == 'mcdr':
            init_kwargs['dropout_rate'] = args.dropout_rate
        elif method == 'vtsr':
            init_kwargs['temperature_mode'] = args.vtsr_mode

        new_router = RouterClass(**init_kwargs).to(device)
        layer.block_sparse_moe.router = new_router

    model.eval()
    return model

def calculate_flops_per_token(model, method, args, device):
    """
    Calculates the effective FLOPs per token for a prediction.
    """
    seq_len = 128
    dummy_input = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

    # Configure model for the correct inference mode (e.g., set num_samples)
    if method in ['mcdr', 'mfvr', 'fcvr']:
        for layer in model.base_model.model.model.layers:
            router = layer.block_sparse_moe.router
            if hasattr(router, 'num_mc_samples_inference'):
                router.num_mc_samples_inference = args.num_samples
    
    # Run FLOPs Analysis
    flop_analyzer = FlopCountAnalysis(model, dummy_input)
    total_gflops = flop_analyzer.total() / 1e9 # Convert to GFLOPs
    
    # Normalize by the number of tokens to get FLOPs per token
    effective_gflops_per_token = total_gflops / seq_len
    
    return {'effective_gflops_per_token': effective_gflops_per_token}

def main():
    """Main orchestration script for FLOPs analysis."""
    setup_environment()
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    methods_to_analyze = ['det', 'mcdr', 'mfvr', 'fcvr', 'vtsr']
    all_results = {}

    for method in methods_to_analyze:
        print(f"\n--- Analyzing FLOPs for method: {method.upper()} ---")

        # Load a fresh copy of the base model for each method to ensure a clean state
        model = load_peft_model_and_adapter(
            args.model_shortcode,
            adapter_path=args.kvq_adapter_path,
            device_map=device
        )
        
        model = prepare_model(model, method, args.map_router_path, args)
        
        flops_results = calculate_flops_per_token(model, method, args, device)
        all_results[method] = flops_results
        
        print(f"  - Effective GFLOPs per Token: {flops_results['effective_gflops_per_token']:.4f}")

    # Save the final aggregated results
    os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)
    with open(args.output_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved all FLOPs analysis results to {args.output_json_path}")

if __name__ == "__main__":
    main()
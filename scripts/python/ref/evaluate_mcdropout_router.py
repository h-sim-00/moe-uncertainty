# evaluate_mcdropout_router.py

import argparse
import os
import torch
import pandas as pd
from utils import setup_environment, load_classification_dataset
from model import load_peft_model_and_adapter, load_tokenizer
from model.routers.mcdropout import MCDropoutRouter, evaluate_mcdropout_router

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned MC Dropout MoE router.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--base_adapter_path", type=str, required=True, help="Path to the Stage 1 KVQ LoRA adapter.")
    parser.add_argument("--router_weights_dir", type=str, required=True, help="Path to the directory containing saved MC Dropout router weights.")
    parser.add_argument("--dropout_rate", type=float, required=True, help="Dropout rate used during training for re-instantiation.")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--results_csv_path", type=str, default="./results/evaluation_mcdropout.csv")
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 1. Load the base model and attach the Stage 1 fine-tuned adapter
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.base_adapter_path,
        device_map="auto"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. "Lego Swap": Replace original routers and load the trained weights
    print(f"Loading trained MC Dropout routers from '{args.router_weights_dir}'")
    causal_model = model.base_model.model.model
    for i, layer in enumerate(causal_model.layers):
        new_router = MCDropoutRouter(
            config=causal_model.config,
            dropout_rate=args.dropout_rate
        )
        weights_path = os.path.join(args.router_weights_dir, f"layer_{i}_weights.pt")
        new_router.load_weights(weights_path, device=model.device)
        new_router.to(model.device)
        layer.block_sparse_moe.router = new_router
    
    print("Model is ready for evaluation.")

    # 3. Run evaluation across all specified datasets
    dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc", "jp", "phi", "pro_law", "abs_alg", "cs", "med_gen"]
    all_results = []
    
    for shortcode in dataset_shortcodes:
        test_raw_dataset = load_classification_dataset(shortcode, split="test")
        results = evaluate_mcdropout_router(
            model, tokenizer, test_raw_dataset,
            dataset_name=shortcode, num_samples=args.num_samples,
            batch_size=args.batch_size
        )
        all_results.append(results)

    # 4. Save results to a CSV
    df = pd.DataFrame(all_results)
    df.to_csv(args.results_csv_path, index=False)
    print(f"\nSaved all evaluation results to {args.results_csv_path}")

if __name__ == "__main__":
    main()
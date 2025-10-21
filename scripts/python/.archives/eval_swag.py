# evaluate_swag_router.py

import argparse
import torch
import pandas as pd
from utils import setup_environment, load_classification_dataset
from model import load_peft_model_and_adapter, load_tokenizer
from model.routers.swag import evaluate_swag_router

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a SWAG-approximated MoE router.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=20)
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 1. Load the base model with the Stage 1 adapter
    base_adapter_path = f"./adapters/kvq_ft_{args.model_shortcode}_seed-{args.seed}"
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=base_adapter_path,
        device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)
    
    # 2. Load the fitted SWAG objects
    swag_path = f"./adapters/swag_{args.model_shortcode}_seed-{args.seed}/swag_routers.pt"
    swag_objects = torch.load(swag_path, map_location=model.device)
    # The SWAG object automatically handles loading its mean weights into the base model

    # 3. Run evaluation across all specified datasets
    dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc", "jp", "phi", "pro_law", "abs_alg", "cs", "med_gen"]
    all_results = []
    
    for shortcode in dataset_shortcodes:
        test_raw_dataset = load_classification_dataset(shortcode, split="test")
        results = evaluate_swag_router(
            model, tokenizer, swag_objects, test_raw_dataset,
            dataset_name=shortcode, 
            num_samples=args.num_samples,
            batch_size=args.batch_size
        )
        all_results.append(results)

    # 4. Save results to a CSV
    results_csv_path = f"./results/eval_swag_{args.model_shortcode}_n-{args.num_samples}_seed-{args.seed}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv_path, index=False)
    print(f"\nSaved all evaluation results to {args.results_csv_path}")

if __name__ == "__main__":
    main()
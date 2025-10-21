# evaluate_svgd_router.py

import argparse
import torch
import pandas as pd
from utils import setup_environment, load_classification_dataset
from model import load_peft_model_and_adapter, load_tokenizer
from model.routers.svgd import evaluate_svgd_router

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned SVGD MoE router.")
    parser.add_argument("--model_shortcode", type=str, default="granite")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def main():
    print("Setting up the environment...")
    setup_environment()
    args = parse_args()

    # 1. Load the base model and attach the Stage 1 adapter
    base_adapter_path = f"./adapters/kvq_ft_{args.model_shortcode}_seed-{args.seed}"
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=base_adapter_path,
        device_map="cuda:0"
    )
    tokenizer = load_tokenizer(args.model_shortcode)

    # 2. Load the trained SVGD particles
    particles_path = f"./adapters/svgd_{args.model_shortcode}_seed-{args.seed}/router_particles.pt"
    print(f"Loading SVGD particles from '{particles_path}'")
    particles = torch.load(particles_path, map_location=model.device)
    
    # 3. Run evaluation across all specified datasets
    dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc", "jp", "phi", "pro_law", "abs_alg", "cs", "med_gen"]
    all_results = []
    
    for shortcode in dataset_shortcodes:
        test_raw_dataset = load_classification_dataset(shortcode, split="test")
        results = evaluate_svgd_router(
            model, tokenizer, particles, test_raw_dataset,
            dataset_name=shortcode, batch_size=args.batch_size
        )
        all_results.append(results)

    # 4. Save results to a CSV
    results_csv_path = f"./results/eval_svgd_{args.model_shortcode}_seed-{args.seed}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv_path, index=False)
    print(f"\nSaved all evaluation results to {results_csv_path}")

if __name__ == "__main__":
    main()
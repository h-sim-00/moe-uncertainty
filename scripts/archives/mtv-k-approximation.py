import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assume these are in your project structure
from model import load_peft_model_and_adapter, load_tokenizer
from utils import load_classification_dataset

# --- Global list to hold hook data ---
captured_metrics = []

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the 'Illusion of Router Confidence' experiment.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--adapter_path", type=str, default="./adapters/kvq_ft_granite_seed-42", help="Path to the saved MAP baseline adapter.")
    parser.add_argument("--output_dir", type=str, default="./ref/mtv-2-10000", help="Directory to save results.")
    parser.add_argument("--num_tokens", type=int, default=10000, help="Number of random tokens to sample for the analysis.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    return parser.parse_args()

def prepare_model_with_hooks(model, top_k):
    """
    Attaches forward hooks to each router to capture the necessary metrics.
    """
    global captured_metrics
    
    def router_metrics_hook(module, input, output):
        # The router's output is a tuple, the last element is the logits
        logits = output[-1].detach() # Shape: [num_tokens_in_batch, num_experts]
        
        # Sort logits to find the boundary
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
        
        # 1. Top-K Boundary Ambiguity (Delta_K)
        logit_k = sorted_logits[:, top_k - 1]
        logit_k_plus_1 = sorted_logits[:, top_k]
        delta_k = logit_k - logit_k_plus_1
        
        # 2. Top-K Gating Confidence (W_K)
        # Note: The router already performs top-k internally, but for this metric,
        # we need the softmax over all logits.
        full_probs = torch.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(full_probs, dim=-1, descending=True)
        w_k = sorted_probs[:, :top_k].sum(dim=-1)
        
        # Store the metrics for each token in the batch
        for i in range(delta_k.shape[0]):
            captured_metrics.append({
                "delta_k": delta_k[i].item(),
                "w_k": w_k[i].item()
            })

    # Attach the hook to every router in the model
    causal_model = model.base_model.model
    for layer in causal_model.model.layers:
        # We need to find the layer index to tag the data correctly
        # This is a bit of a workaround to pass the layer_idx to the hook
        def create_hook_fn(idx):
            def hook(module, input, output):
                logits = output[-1].detach()
                sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
                delta_k = sorted_logits[:, top_k - 1] - sorted_logits[:, top_k]
                full_probs = torch.softmax(logits, dim=-1)
                sorted_probs, _ = torch.sort(full_probs, dim=-1, descending=True)
                w_k = sorted_probs[:, :top_k].sum(dim=-1)
                for i in range(delta_k.shape[0]):
                    captured_metrics.append({
                        "layer": idx,
                        "delta_k": delta_k[i].item(),
                        "w_k": w_k[i].item()
                    })
            return hook
        
        layer_idx = len(captured_metrics) // (len(causal_model.model.layers) * model.config.max_position_embeddings) if captured_metrics else 0
        
        # A simpler way is to just know the layer index when attaching
        for i, layer in enumerate(causal_model.model.layers):
            layer.block_sparse_moe.router.register_forward_hook(create_hook_fn(i))

    print("Attached metrics hooks to all routers.")
    return model

def collect_router_metrics(model, tokenizer, dataset, num_tokens, batch_size, device):
    """
    Runs inference on a large sample of tokens and collects router metrics via hooks.
    """
    global captured_metrics
    captured_metrics.clear()

    # Create a large flat list of tokens from the dataset
    all_input_ids = []
    for item in dataset:
        # Assuming the dataset format is a dictionary with 'text'
        tokens = tokenizer(item['question'], return_tensors="pt").input_ids.squeeze()
        all_input_ids.append(tokens)
    
    all_input_ids = torch.cat(all_input_ids)
    
    # Sample the required number of tokens
    indices = torch.randperm(len(all_input_ids))[:num_tokens]
    sampled_tokens = all_input_ids[indices]

    print(f"\n--- Running Inference to Collect Metrics on {num_tokens} Tokens ---")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, num_tokens, batch_size), desc="Collecting Metrics"):
            batch_ids = sampled_tokens[i:i+batch_size].unsqueeze(0).to(device)
            model(batch_ids)
            
    return pd.DataFrame(captured_metrics)

def analyze_and_visualize_metrics(df_metrics, output_dir, num_layers):
    """
    Performs quantitative analysis and generates the 2D density plot.
    """
    print("\n--- Analyzing and Visualizing Metrics ---")
    
    # --- Quantitative Analysis ---
    ambiguity_threshold = 0.1
    confidence_threshold = 0.99
    
    analysis_results = []
    for layer_idx in range(num_layers):
        layer_df = df_metrics[df_metrics['layer'] == layer_idx]
        problematic_tokens = layer_df[
            (layer_df['delta_k'] < ambiguity_threshold) & 
            (layer_df['w_k'] > confidence_threshold)
        ]
        percentage = 100 * len(problematic_tokens) / len(layer_df) if len(layer_df) > 0 else 0
        analysis_results.append({"layer": layer_idx, "problematic_percentage": percentage})
        print(f"Layer {layer_idx}: {percentage:.2f}% of tokens are highly ambiguous yet highly confident.")
        
    df_analysis = pd.DataFrame(analysis_results)
    df_analysis.to_csv(os.path.join(output_dir, "quantitative_analysis.csv"), index=False)

    # --- Visualization ---
    grid_rows = 4
    grid_cols = (num_layers + grid_rows - 1) // grid_rows
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 4, grid_rows * 3.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_df = df_metrics[df_metrics['layer'] == layer_idx]
        
        print(f"Visualizing Layer {layer_idx} with {len(layer_df)} tokens.")

        # Use a 2D histogram for the heatmap
        ax.hist2d(
            layer_df["delta_k"], layer_df["w_k"],
            bins=50, #range=[[-0.5, 2.0], [0.5, 1.0]],
            # cmap="viridis"
        )
        
        ax.set_title(f"Layer {layer_idx}")
        # ax.set_xlim(left=-0.5, right=2.0) # Give some space for viewing
        # ax.set_ylim(bottom=0.9, top=1.01) # Focus on the high-confidence region
        ax.axvline(ambiguity_threshold, color='r', linestyle='--', linewidth=1)
        ax.axhline(confidence_threshold, color='r', linestyle='--', linewidth=1)

        print(f"Layer {layer_idx} Done.")

    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)

    # Add a colorbar to the right of the subplots
    # fig.colorbar(
    #     axes[0].collections[0],
    #     ax=axes,
    #     location='right',
    #     pad=0.02,
    #     label='Token Count'
    # )

    fig.supxlabel("Top-K Boundary Ambiguity (Δ_K = logit_K - logit_K+1)", y=0.05)
    fig.supylabel("Top-K Gating Confidence (W_K)", x=-0.05)
    fig.suptitle("The Illusion of Router Confidence: Ambiguity vs. Confidence", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, "router_confidence_heatmap.pdf")
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved visualization to {save_path}")
    plt.close()

def main():
    """Orchestrates the entire experiment."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_peft_model_and_adapter(
        args.model_shortcode,
        adapter_path=args.adapter_path,
        device_map=device
    )
    print(model)
    tokenizer = load_tokenizer(args.model_shortcode)
    num_layers = len(model.base_model.model.model.layers)
    top_k = model.config.num_experts_per_tok
    
    # --- Stage 1: Data Collection ---
    metrics_path = os.path.join(args.output_dir, "router_metrics.csv")
    if not os.path.exists(metrics_path):
        model = prepare_model_with_hooks(model, top_k)
        # Use the ID test sets for data collection
        id_dataset_shortcodes = ["hs_us_his", "hs_gp", "hs_psy", "soc"]
        full_id_dataset = []
        for code in id_dataset_shortcodes:
            full_id_dataset.extend(load_classification_dataset(code, split="test"))
        
        df_metrics = collect_router_metrics(model, tokenizer, full_id_dataset, args.num_tokens, args.batch_size, device)
        df_metrics.to_csv(metrics_path, index=False)
        print(f"Saved collected metrics to {metrics_path}")
    else:
        df_metrics = pd.read_csv(metrics_path)
        print(f"Loaded existing metrics from {metrics_path}")

    # --- Stage 2: Analysis and Visualization ---
    analyze_and_visualize_metrics(df_metrics, args.output_dir, num_layers)

if __name__ == "__main__":
    main()

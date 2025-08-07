import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil

# Assume these are in your project structure
from model import load_peft_model_and_adapter, load_tokenizer

# --- Global list to hold hook data ---
captured_logits = []

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the refined router brittleness experiment.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--adapter_path", type=str, default="./adapters/kvq_ft_granite_seed-42", help="Path to the saved MAP baseline adapter.")
    parser.add_argument("--sentences_csv", type=str, default="ref/sentences.csv", help="Path to the CSV file with sentences.")
    parser.add_argument("--output_dir", type=str, default="./ref/mtv-1", help="Directory to save all results.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--gamma_to_visualize", type=float, default=0.01, help="The gamma value to use for the detailed distributional plot.")
    return parser.parse_args()

def calibrate_and_prepare(model, tokenizer, sentences, device):
    """Calculates L_bar and attaches hooks to the model's routers."""
    print("--- Stage 1: Calibrating Noise Level & Preparing Model ---")
    
    # --- Calibrate Noise ---
    all_norms = []
    embed_layer = model.base_model.model.model.embed_tokens
    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Calibrating Noise"):
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            embeddings = embed_layer(inputs.input_ids)
            norms = torch.linalg.norm(embeddings.view(-1, embeddings.shape[-1]).float(), ord=2, dim=1)
            all_norms.append(norms.cpu())
    l_bar = torch.cat(all_norms).mean().item()
    print(f"Average L2 Norm of Embeddings (L_bar): {l_bar:.4f}")

    # --- Attach Hooks ---
    def router_capture_hook(module, input, output):
        captured_logits.append(output[-1].detach().cpu())

    for layer in model.base_model.model.model.layers:
        layer.block_sparse_moe.router.register_forward_hook(router_capture_hook)
    print("Attached capture hooks to all routers.")
    
    return l_bar, model

def collect_routing_data(model, tokenizer, sentences, l_bar, gammas, batch_size, device, output_dir):
    """Runs inference for each noise level and saves the raw logits."""
    global captured_logits
    
    print("\n--- Stage 2: Collecting Raw Routing Data ---")
    
    # This hook will be attached temporarily to add noise
    perturbation_noise = None
    def embedding_perturb_hook(module, input, output):
        if perturbation_noise is not None:
            noise = perturbation_noise[:output.shape[1], :].to(output.device)
            return output + noise.unsqueeze(0)
        return output
    
    hook_handle = model.base_model.model.model.embed_tokens.register_forward_hook(embedding_perturb_hook)

    for gamma in gammas:
        sigma = gamma * l_bar
        gamma_str = f"{gamma:.3f}".replace('.', 'p')
        output_path = os.path.join(output_dir, f"raw_logits_gamma_{gamma_str}.pt")
        
        if os.path.exists(output_path):
            print(f"Data for gamma={gamma:.3f} already exists. Skipping.")
            continue

        print(f"\nCollecting data for gamma = {gamma:.3f} (sigma = {sigma:.6f})")
        
        # --- Original Pass ---
        perturbation_noise = None
        captured_logits.clear()
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc=f"Original Pass (gamma={gamma:.3f})"):
                batch = sentences[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                model(inputs.input_ids)
        original_logits = list(captured_logits)

        # --- Perturbed Pass ---
        max_len = 2048
        embed_dim = model.config.hidden_size
        perturbation_noise = torch.randn(max_len, embed_dim) * sigma
        captured_logits.clear()
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc=f"Perturbed Pass (gamma={gamma:.3f})"):
                batch = sentences[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                model(inputs.input_ids)
        perturbed_logits = list(captured_logits)
        
        torch.save({
            "gamma": gamma,
            "sigma": sigma,
            "original": original_logits,
            "perturbed": perturbed_logits
        }, output_path)
        print(f"Saved data to {output_path}")

    hook_handle.remove() # Clean up the embedding hook

def compute_jaccard_scores(output_dir, top_k, num_layers):
    """Processes raw logits files to compute and save Jaccard scores."""
    print("\n--- Stage 3: Computing Jaccard Scores ---")
    all_scores = []
    
    data_files = [f for f in os.listdir(output_dir) if f.startswith("raw_logits_gamma_") and f.endswith(".pt")]
    
    for filename in tqdm(data_files, desc="Processing Raw Data Files"):
        data = torch.load(os.path.join(output_dir, filename))
        gamma = data['gamma']
        
        num_batches = len(data['original']) // num_layers
        
        original_by_layer = [torch.cat([data['original'][i + l*num_batches] for i in range(num_batches)]) for l in range(num_layers)]
        perturbed_by_layer = [torch.cat([data['perturbed'][i + l*num_batches] for i in range(num_batches)]) for l in range(num_layers)]
        
        for layer_idx in range(num_layers):
            orig_indices = torch.topk(original_by_layer[layer_idx], k=top_k, dim=-1).indices
            pert_indices = torch.topk(perturbed_by_layer[layer_idx], k=top_k, dim=-1).indices
            
            for token_orig, token_pert in zip(orig_indices, pert_indices):
                set_orig = set(token_orig.tolist())
                set_pert = set(token_pert.tolist())
                intersection = len(set_orig.intersection(set_pert))
                union = len(set_orig.union(set_pert))
                score = intersection / union if union > 0 else 0
                all_scores.append({"gamma": gamma, "layer": layer_idx, "jaccard_score": score})
                
    return pd.DataFrame(all_scores)

def prepare_data_for_visuals(df_scores, output_dir, gamma_to_visualize):
    """Creates and saves the specific data files needed for plotting."""
    print("\n--- Stage 4: Preparing Data for Visualization ---")
    
    # 1. Data for Sensitivity Plot
    sensitivity_data = df_scores.groupby(['gamma', 'layer'])['jaccard_score'].mean().reset_index()
    sensitivity_path = os.path.join(output_dir, "sensitivity_plot_data.csv")
    sensitivity_data.to_csv(sensitivity_path, index=False)
    print(f"Saved sensitivity plot data to {sensitivity_path}")

    # 2. Data for Distributional Plot
    distributional_data = df_scores[df_scores['gamma'] == gamma_to_visualize]
    dist_path = os.path.join(output_dir, f"distributional_plot_data_gamma_{str(gamma_to_visualize).replace('.', 'p')}.csv")
    distributional_data.to_csv(dist_path, index=False)
    print(f"Saved distributional plot data to {dist_path}")

    return sensitivity_path, dist_path

def generate_visualizations(sensitivity_data_path, distributional_data_path, output_dir, num_layers):
    """Generates and saves the two final plots."""
    print("\n--- Stage 5: Generating Visualizations ---")
    
    # --- Plot 1: Sensitivity Analysis ---
    df_sens = pd.read_csv(sensitivity_data_path)
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_sens, x='layer', y='jaccard_score', hue='gamma', marker='o', palette='viridis')
    plt.title("Router Stability Across Layers and Noise Levels", fontsize=16)
    plt.xlabel("MoE Layer")
    plt.ylabel("Mean Jaccard Similarity")
    plt.xticks(np.arange(0, num_layers, 2))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Noise γ (x L_bar)")
    save_path_sens = os.path.join(output_dir, "sensitivity_analysis.png")
    plt.savefig(save_path_sens, dpi=300)
    print(f"Saved sensitivity analysis plot to {save_path_sens}")
    plt.close()

    # --- Plot 2: Distributional Analysis (Vertical Density Plots) ---
    df_dist = pd.read_csv(distributional_data_path)
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 0.8, 8), sharey=True)
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_scores = df_dist[df_dist['layer'] == layer_idx]['jaccard_score']
        
        sns.kdeplot(y=layer_scores, ax=ax, color="C0", fill=True)
        
        ax.set_ylim(0, 1)
        ax.set_xticks([]) # Hide x-axis ticks
        ax.set_xlabel(f"{layer_idx}", fontsize=10) # Use x-label for layer number
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    axes[0].set_ylabel("Jaccard Similarity Score")
    fig.suptitle(f"Distribution of Router Stability (Noise γ = {df_dist['gamma'].iloc[0]:.3f})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path_dist = os.path.join(output_dir, f"distributional_analysis.pdf")
    plt.savefig(save_path_dist, dpi=300)
    print(f"Saved distributional analysis plot to {save_path_dist}")
    plt.close()

def main():
    """Orchestrates the entire experiment."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_peft_model_and_adapter(args.model_shortcode, adapter_path=args.adapter_path, device_map=device)
    model.eval()
    tokenizer = load_tokenizer(args.model_shortcode)
    df_sentences = pd.read_csv(args.sentences_csv)
    sentences = df_sentences['sentence'].tolist()
    num_layers = len(model.base_model.model.model.layers)
    top_k = model.config.num_experts_per_tok
    gammas = [0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05]
    
    # --- Stage 1: Calibrate & Prepare ---
    l_bar, model = calibrate_and_prepare(model, tokenizer, sentences, device)
    
    # --- Stage 2: Collect Data ---
    collect_routing_data(model, tokenizer, sentences, l_bar, gammas, args.batch_size, device, args.output_dir)
    
    # --- Stage 3: Compute Scores ---
    jaccard_scores_path = os.path.join(args.output_dir, "all_jaccard_scores.csv")
    if not os.path.exists(jaccard_scores_path):
        df_scores = compute_jaccard_scores(args.output_dir, top_k, num_layers)
        df_scores.to_csv(jaccard_scores_path, index=False)
    else:
        df_scores = pd.read_csv(jaccard_scores_path)
        print(f"\nLoaded existing Jaccard scores from {jaccard_scores_path}")

    # --- Stage 4: Prepare Visualization Data ---
    sens_path, dist_path = prepare_data_for_visuals(df_scores, args.output_dir, args.gamma_to_visualize)

    # --- Stage 5: Generate Visualizations ---
    generate_visualizations(sens_path, dist_path, args.output_dir, num_layers)

    print("\nExperiment complete.")

if __name__ == "__main__":
    main()

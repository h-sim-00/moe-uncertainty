import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Assume these are in your project structure
from model import load_model, load_tokenizer

# --- Global variables to hold hook data ---
# This is a simple way to manage state for the hooks.
# A class-based approach would also work well.
captured_expert_indices = []
perturbation_noise = None

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the MoE router instability experiment.")
    parser.add_argument("--model_shortcode", type=str, default="granite", help="Shortcode for the model to use.")
    parser.add_argument("--sentences_csv", type=str, default="./ref/sentences.csv", help="Path to the CSV file containing sentences.")
    parser.add_argument("--output_dir", type=str, default="./ref/mtv-1-01", help="Directory to save intermediate and final results.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    return parser.parse_args()

def calibrate_noise_level(model, tokenizer, sentences, device):
    """
    Calculates the appropriate noise level (sigma) based on the average
    L2 norm of the token embeddings.
    """
    print("--- Calibrating Noise Level ---")
    all_norms = []
    embed_layer = model.model.embed_tokens
    
    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Calculating Embedding Norms"):
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            embeddings = embed_layer(inputs.input_ids)
            # Flatten to get a list of all token embeddings in the batch
            embeddings_flat = embeddings.view(-1, embeddings.shape[-1])
            norms = torch.linalg.norm(embeddings_flat.float(), ord=2, dim=1)
            all_norms.append(norms.cpu())
            
    all_norms_tensor = torch.cat(all_norms)
    l_bar = all_norms_tensor.mean().item()
    sigma = 0.01 * l_bar
    
    print(f"Average L2 Norm of Embeddings (L_bar): {l_bar:.4f}")
    print(f"Calculated Noise Sigma (0.01 * L_bar): {sigma:.6f}")
    return sigma

def prepare_model_with_hooks(model, top_k):
    """
    Attaches forward hooks to the model to perturb embeddings and capture router decisions.
    """
    global captured_expert_indices, perturbation_noise

    # Hook 1: Perturbs the output of the embedding layer
    def embedding_perturb_hook(module, input, output):
        if perturbation_noise is not None:
            # Ensure noise is the same shape as the output embeddings
            noise = perturbation_noise[:output.shape[1], :].to(output.device)
            return output + noise.unsqueeze(0)
        return output

    # Hook 2: Captures the top-k expert indices from the router's output logits
    def router_capture_hook(module, input, output):
        # The router's output is a tuple, the last element is the logits
        logits = output[-1]
        # We re-compute topk here to get the indices
        top_k_indices = torch.topk(logits, k=top_k, dim=-1).indices
        captured_expert_indices.append(top_k_indices.detach().cpu())

    # Attach hooks
    model.model.embed_tokens.register_forward_hook(embedding_perturb_hook)
    for layer in model.model.layers:
        layer.block_sparse_moe.router.register_forward_hook(router_capture_hook)
        
    print("Attached hooks to embedding layer and all routers.")
    return model

def run_inference_and_collect_data(model, tokenizer, sentences, sigma, batch_size, device, output_dir):
    """
    Runs inference twice (original and perturbed) and saves the captured expert indices.
    """
    global captured_expert_indices, perturbation_noise
    
    original_indices_path = os.path.join(output_dir, "original_indices.pt")
    perturbed_indices_path = os.path.join(output_dir, "perturbed_indices.pt")

    # --- 1. Original Run (No Perturbation) ---
    print("\n--- Running Inference on Original Sentences ---")
    perturbation_noise = None
    captured_expert_indices.clear()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Original Inference"):
            batch = sentences[i:i+batch_size]
            tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            model(tokenizer(batch, return_tensors="pt", padding=True, truncation=True).input_ids.to(device))

    torch.save(captured_expert_indices, original_indices_path)
    print(f"Saved original expert indices to {original_indices_path}")

    # --- 2. Perturbed Run ---
    print("\n--- Running Inference on Perturbed Sentences ---")
    # Generate a large noise tensor once to be sliced by the hook
    # Max sequence length and embedding dim
    max_len = 2048 
    embed_dim = model.config.hidden_size
    perturbation_noise = torch.randn(max_len, embed_dim) * sigma
    captured_expert_indices.clear()

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Perturbed Inference"):
            batch = sentences[i:i+batch_size]
            model(tokenizer(batch, return_tensors="pt", padding=True, truncation=True).input_ids.to(device))

    torch.save(captured_expert_indices, perturbed_indices_path)
    print(f"Saved perturbed expert indices to {perturbed_indices_path}")
    
    return original_indices_path, perturbed_indices_path

def compute_jaccard_scores(original_indices_path, perturbed_indices_path, num_layers):
    """
    Loads the raw data and computes the Jaccard similarity for each token at each layer.
    """
    print("\n--- Computing Jaccard Scores ---")
    original_data = torch.load(original_indices_path)
    perturbed_data = torch.load(perturbed_indices_path)

    # The data is a flat list of tensors: [layer0_b0, layer1_b0, ..., layerL_b0, layer0_b1, ...]
    # We need to un-flatten it first.
    num_batches = len(original_data) // num_layers
    original_by_layer = [torch.cat([original_data[i + l*num_batches] for i in range(num_batches)]) for l in range(num_layers)]
    perturbed_by_layer = [torch.cat([perturbed_data[i + l*num_batches] for i in range(num_batches)]) for l in range(num_layers)]

    all_scores = []
    for layer_idx in tqdm(range(num_layers), desc="Calculating Jaccard Similarity"):
        layer_original = original_by_layer[layer_idx]
        layer_perturbed = perturbed_by_layer[layer_idx]
        
        for token_orig, token_pert in zip(layer_original, layer_perturbed):
            set_orig = set(token_orig.tolist())
            set_pert = set(token_pert.tolist())
            
            intersection = len(set_orig.intersection(set_pert))
            union = len(set_orig.union(set_pert))
            
            score = intersection / union if union > 0 else 0
            all_scores.append({"layer": layer_idx, "jaccard_score": score})
            
    return pd.DataFrame(all_scores)

def visualize_scores(df, output_dir, num_layers):
    """
    Generates and saves the grid of histograms for Jaccard scores.
    """
    print("\n--- Generating Visualization ---")
    # Determine grid size (e.g., 4x8 for 32 layers)
    grid_rows = 4
    grid_cols = (num_layers + grid_rows - 1) // grid_rows # Ceiling division

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 4, grid_rows * 3), sharex=True, sharey=True)
    axes = axes.flatten() # Flatten to make iteration easier

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_scores = df[df['layer'] == layer_idx]['jaccard_score']
        
        ax.hist(layer_scores, bins=np.linspace(0, 1, 11), density=True, color='skyblue', edgecolor='black')
        
        mean_score = layer_scores.mean()
        ax.axvline(mean_score, color='r', linestyle='--', linewidth=2)
        ax.text(0.05, 0.9, f"μ = {mean_score:.2f}", transform=ax.transAxes, ha='left', va='top', color='r')
        
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlim(0, 1)

    # Hide unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)

    fig.supxlabel("Jaccard Similarity Score")
    fig.supylabel("Density")
    fig.suptitle("Distribution of Router Stability Under Input Perturbation", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, "jaccard_histograms.pdf")
    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization to {save_path}")
    plt.close()

def main():
    """Orchestrates the entire experiment."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_shortcode, device_map=device)
    model.eval() # Ensure model is in eval mode
    tokenizer = load_tokenizer(args.model_shortcode)
    df = pd.read_csv(args.sentences_csv)
    sentences = df['sentence'].tolist()
    num_layers = len(model.model.layers)
    top_k = model.config.num_experts_per_tok
    
    # --- Stage 1: Calibrate Noise ---
    sigma_path = os.path.join(args.output_dir, "sigma.pt")
    if not os.path.exists(sigma_path):
        sigma = calibrate_noise_level(model, tokenizer, sentences, device)
        torch.save(sigma, sigma_path)
    else:
        sigma = torch.load(sigma_path)
        print(f"Loaded existing sigma: {sigma:.6f}")

    # --- Stage 2: Run Inference ---
    original_indices_path = os.path.join(args.output_dir, "original_indices.pt")
    perturbed_indices_path = os.path.join(args.output_dir, "perturbed_indices.pt")
    if not os.path.exists(original_indices_path) or not os.path.exists(perturbed_indices_path):
        model = prepare_model_with_hooks(model, top_k)
        run_inference_and_collect_data(model, tokenizer, sentences, sigma, args.batch_size, device, args.output_dir)
    else:
        print("Found existing inference data. Skipping inference.")

    # --- Stage 3: Compute Metrics ---
    jaccard_scores_path = os.path.join(args.output_dir, "jaccard_scores.csv")
    if not os.path.exists(jaccard_scores_path):
        df_scores = compute_jaccard_scores(original_indices_path, perturbed_indices_path, num_layers)
        df_scores.to_csv(jaccard_scores_path, index=False)
        print(f"Saved Jaccard scores to {jaccard_scores_path}")
    else:
        df_scores = pd.read_csv(jaccard_scores_path)
        print(f"Loaded existing Jaccard scores from {jaccard_scores_path}")

    # --- Stage 4: Visualize ---
    visualize_scores(df_scores, args.output_dir, num_layers)

if __name__ == "__main__":
    main()

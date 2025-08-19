import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import json
import os

def plot_sensitivity_analysis(df, output_path):
    """
    Generates the scattered line graph for sensitivity analysis across noise levels.
    """
    print("Generating sensitivity analysis plot...")
    
    # Calculate the mean Jaccard score for each group
    sensitivity_data = df.groupby(['gamma', 'layer'])['jaccard_score'].mean().reset_index()

    plt.figure(figsize=(16, 8))
    # Format gamma values for consistent legend labels
    sensitivity_data['gamma_str'] = sensitivity_data['gamma'].apply(lambda g: f"{g:.4g}")
    sns.lineplot(
        data=sensitivity_data, 
        x='layer', 
        y='jaccard_score', 
        hue='gamma_str', 
        marker='o', 
        palette='viridis_r' # Reversed viridis is often intuitive for "higher is better"
    )
    
    plt.title("Router Stability Across Layers and Noise Levels", fontsize=14, pad=20)
    plt.xlabel("MoE Layer", fontsize=12)
    plt.ylabel("Mean Jaccard Similarity", fontsize=12)
    plt.xticks(np.arange(0, 32, 2))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Noise γ", loc='lower left')
    plt.tight_layout()  # Adjust layout to fit the title
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved sensitivity analysis plot to '{output_path}'")
    plt.close()

def plot_distributional_analysis(df, gamma_to_visualize, output_path, num_layers=32):
    """
    Generates the vertical, side-by-side density plots for a fixed noise level.
    Adds a mean value marker to each plot and customizes x-axis labeling.
    Minimizes whitespace around the figure.
    """
    print(f"Generating distributional analysis plot for gamma = {gamma_to_visualize}...")
    
    # Filter the data for the chosen gamma
    dist_data = df[df['gamma'] == gamma_to_visualize]
    
    if dist_data.empty:
        print(f"Warning: No data found for gamma = {gamma_to_visualize}. Skipping plot.")
        return

    fig, axes = plt.subplots(1, num_layers, figsize=(16, 8), sharey=True)
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_scores = dist_data[dist_data['layer'] == layer_idx]['jaccard_score']
        
        if not layer_scores.empty:
            # Plot KDE
            sns.kdeplot(y=layer_scores, ax=ax, color="C0", fill=True, linewidth=1.5)
            # Add mean marker
            mean_val = layer_scores.mean()
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
            ax.scatter(0, mean_val, color='red', s=30, zorder=5)
        
        # --- Aesthetics and Formatting ---
        ax.set_ylim(0, 1)
        ax.set_xticks([]) # Hide x-axis ticks (density values)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Use a clean title for the layer number below the plot
        ax.set_title(f"{layer_idx}", y=-0.05, fontsize=10)
        ax.set_xlabel("", fontsize=9, labelpad=10)

    # Set labels for the first subplot only
    axes[0].set_ylabel("Jaccard Similarity Score", fontsize=12)
    
    # Add a horizontal baseline line at y=0.6 across all layers
    for ax in axes:
        ax.axhline(0.5, color='green', linestyle=':', linewidth=2, label='Baseline (0.5)')
    
    # Add a single legend for the mean line (red dashed)
    handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean Value')]
    handles.append(plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2, label='Baseline (0.6)'))
    fig.legend(handles=handles, loc='upper right', fontsize=12)

    fig.suptitle(f"Distribution of Router Stability (Noise γ = {gamma_to_visualize})", fontsize=14, y=0.96)
    fig.supxlabel("MoE Layer", fontsize=12, y=0.03)

    # Minimize whitespace around the figure
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.01, right=0.99, wspace=0.05)
    # plt.tight_layout(rect=[3, 0, 1, 1])  # Adjust layout to fit the title
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved distributional analysis plot to '{output_path}'")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize router stability results.")
    parser.add_argument("--input_json", type=str, default="new_jaccard_scores.json", help="Path to the input JSON file with Jaccard scores.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save the plots.")
    parser.add_argument("--gamma_to_visualize", type=float, default=0.01, help="The gamma value for the detailed distributional plot.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data from the JSON file
    print(f"Loading data from '{args.input_json}'...")
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    df_scores = pd.DataFrame(data)
    
    # Generate the plots
    sensitivity_plot_path = os.path.join(args.output_dir, "mtv1_new_sensitivity_analysis.pdf")
    plot_sensitivity_analysis(df_scores, sensitivity_plot_path)
    
    distributional_plot_path = os.path.join(args.output_dir, f"mtv1_new_distributional_analysis_gamma_{str(args.gamma_to_visualize).replace('.', 'p')}.pdf")
    plot_distributional_analysis(df_scores, args.gamma_to_visualize, distributional_plot_path)

if __name__ == "__main__":
    main()
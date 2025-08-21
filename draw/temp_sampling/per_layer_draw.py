import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os
import re
from tqdm import tqdm

def aggregate_ts_per_layer_data(results_dir):
    """
    Aggregates all deterministic and per-layer temperature sampling results
    into a single pandas DataFrame.
    """
    print("--- Aggregating experiment data ---")
    all_results = []

    # --- 1. Process Deterministic Baselines ---
    det_dir = os.path.join(results_dir, "det")
    for filename in os.listdir(det_dir):
        if filename.startswith("id_calib_det_granite_") and filename.endswith(".json"):
            # Extract dataset name from filename, e.g., 'obqa'
            dataset_name = filename[21:].replace('.json', '')
            
            with open(os.path.join(det_dir, filename), 'r') as f:
                data = json.load(f)
            
            # The JSON has keys for each dataset, we only want the one matching the file
            if dataset_name in data:
                metrics = data[dataset_name]
                for metric_name, value in metrics.items():
                    all_results.append({
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "type": "baseline",
                        "value": value
                    })

    # --- 2. Process Temperature Sampling Results ---
    ts_dir = os.path.join(results_dir, "temp_sampling", "per_layer")
    
    # Regex to parse the filename, e.g., id_calib_granite_medmcqa_med_layer-24_temp-0.5.json
    pattern = re.compile(r"id_calib_granite_(.*?)_layer-(\d+)_temp-([\d.]+)\.json")

    for filename in tqdm(os.listdir(ts_dir), desc="Parsing TS files"):
        match = pattern.match(filename)
        if match:
            dataset_name, layer, temp = match.groups()
            layer = int(layer)
            temp = float(temp)
            
            with open(os.path.join(ts_dir, filename), 'r') as f:
                data = json.load(f)
            
            # The JSON has a single dictionary of metrics
            if dataset_name in data:
                metrics = data[dataset_name]
                for metric_name, value in metrics.items():
                    # We only care about the metrics, not the dataset name inside the file
                    if metric_name.upper() in ["ACC", "NLL", "ECE", "MCE"]:
                        all_results.append({
                            "dataset": dataset_name,
                            "metric": metric_name.upper(),
                            "type": "temp_sampling",
                            "layer": layer,
                            "temperature": temp,
                            "value": value
                        })

    if not all_results:
        raise FileNotFoundError("No result files found. Check your results directory structure.")

    return pd.DataFrame(all_results)

def plot_ts_per_layer_grid(df, output_image_path):
    """
    Generates the 3x4 grid plot from the aggregated data.
    """
    print("\n--- Generating Visualization ---")
    
    # Separate baselines from the main data
    df_baselines = df[df['type'] == 'baseline'].set_index(['dataset', 'metric'])['value']
    df_ts = df[df['type'] == 'temp_sampling'].copy()

    datasets = ["obqa", "sciq", "medmcqa_med"]
    metrics = ["ACC", "NLL", "ECE", "MCE"]
    
    fig, axes = plt.subplots(len(datasets), len(metrics), 
                             figsize=(24, 16), 
                             sharex=True,
                             gridspec_kw={'wspace': 0.15, 'hspace': 0.1})

    # fig.suptitle("Impact of Per-Layer Temperature Sampling on ID Calibration", fontsize=24, y=0.94)

    for row_idx, dataset in enumerate(datasets):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            # --- 1. Plot Deterministic Baseline ---
            baseline_value = df_baselines.get((dataset, metric), None)
            if baseline_value is not None:
                ax.axhline(y=baseline_value, color='red', linestyle='--', linewidth=2, label='Deterministic Baseline')

            # --- 2. Plot Temperature Sampling Lines ---
            subplot_data = df_ts[(df_ts['dataset'] == dataset) & (df_ts['metric'] == metric)]
            
            if not subplot_data.empty:
                sns.lineplot(
                    data=subplot_data,
                    x='layer',
                    y='value',
                    hue='temperature',
                    ax=ax,
                    marker='o',
                    palette='viridis'
                )

            # --- Aesthetics ---
            if row_idx == 0:
                ax.set_title(metric, fontsize=18, pad=15)
            if col_idx == 0:
                ax.set_ylabel(dataset, fontsize=18, labelpad=15)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("MoE Layer", fontsize=14)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Handle legends: show only one
            if not (row_idx == 0 and col_idx == len(metrics) - 1):
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            else:
                ax.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')

    # plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Saved final plot to '{output_image_path}'")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize per-layer temperature sampling results.")
    parser.add_argument("--results_dir", type=str, default="./results", help="Root directory containing the 'det' and 'temp_sampling' subfolders.")
    parser.add_argument("--output_image", type=str, default="./draw/temp_sampling/ts_per_layer_results.pdf", help="Path to save the output plot.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_image), exist_ok=True)

    # 1. Aggregate data from all JSON files
    df_aggregated = aggregate_ts_per_layer_data(args.results_dir)
    
    # Optional: Save the aggregated data for future use
    df_aggregated.to_csv("./draw/temp_sampling/aggregated_ts_per_layer_data.csv", index=False)
    print("Saved aggregated data to aggregated_ts_per_layer_data.csv")
    
    # 2. Generate the plot
    plot_ts_per_layer_grid(df_aggregated, args.output_image)

if __name__ == "__main__":
    main()
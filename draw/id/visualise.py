import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import numpy as np
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# def plot_calibration_grid(input_json_path, output_image_path):
#     """
#     Loads calibration data and generates the original 3x4 grid plot
#     with grouped bars and x-axis labels.
#     """
#     print(f"Loading data from '{input_json_path}'...")
#     with open(input_json_path, 'r') as f:
#         data = json.load(f)
#     df = pd.DataFrame(data)

#     datasets = sorted(df['dataset'].unique())
#     metrics = ["ACC", "NLL", "ECE", "MCE"]
    
#     fig, axes = plt.subplots(len(datasets), len(metrics), 
#                              figsize=(24, 16), 
#                             #  sharey='row',
#                              gridspec_kw={'wspace': 0.1, 'hspace': 0.45})

#     # fig.suptitle("ID Calibration Performance Across Bayesian Methods (Detailed View)", fontsize=24, y=0.97)

#     for row_idx, dataset in enumerate(datasets):
#         for col_idx, metric in enumerate(metrics):
#             ax = axes[row_idx, col_idx]
            
#             subplot_data = df[(df['dataset'] == dataset) & (df['metric'] == metric)]
            
#             # --- CORRECTED LOGIC: Part 1 ---
#             # Define a consistent order for methods and hues
#             methods_order = sorted(subplot_data['method'].unique())
#             hue_order = sorted(subplot_data['layer_selection'].unique())
            
#             sns.barplot(
#                 data=subplot_data, x='method', y='mean_value', hue='layer_selection',
#                 ax=ax, palette='viridis', edgecolor='black',
#                 order=methods_order, # Force seaborn to use our defined order
#                 hue_order=hue_order  # Force the hue order as well
#             )
            
#             # --- CORRECTED LOGIC: Part 2 ---
#             # Now, use the same defined order to plot error bars
#             x_positions = np.arange(len(methods_order))
#             bar_width = 0.4 
            
#             for i, method in enumerate(methods_order): # Iterate using the defined order
#                 for j, hue in enumerate(hue_order):   # Iterate using the defined order
#                     position = x_positions[i] + (j - 0.5) * bar_width
                    
#                     data_point = subplot_data[
#                         (subplot_data['method'] == method) &
#                         (subplot_data['layer_selection'] == hue)
#                     ]
                    
#                     if not data_point.empty:
#                         mean = data_point['mean_value'].iloc[0]
#                         std_dev = data_point['std_dev'].iloc[0]
#                         ax.errorbar(x=position, y=mean, yerr=std_dev, fmt='none', c='black', capsize=3)

#             # ... (Aesthetics remain the same) ...
#             if row_idx == 0:
#                 ax.set_title(metric_labels[metric], fontsize=18, pad=15)
#             if col_idx == 0:
#                 ax.set_ylabel(dataset, fontsize=18, labelpad=15)
#             else:
#                 ax.set_ylabel("")

#             ax.tick_params(axis='x', rotation=65, labelsize=12)
#             ax.set_xlabel("")
#             ax.grid(axis='y', linestyle='--', alpha=0.7)
            
#             if row_idx == 0 and col_idx == len(metrics) - 1:
#                 ax.legend(title='Layer Selection', bbox_to_anchor=(1.05, 1), loc='upper left')
#             else:
#                 if ax.get_legend() is not None:
#                     ax.get_legend().remove()

#     plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
#     plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
#     print(f"Saved detailed plot to '{output_image_path}'")
#     plt.close()

def plot_selected_layers_legend_only(input_json_path, output_image_path):
    """
    Generates a 3x4 grid plot for "Selected Layers", with baselines as bars
    at the start, no error bars for baselines, and visual group separators.
    """
    print(f"Loading data from '{input_json_path}' for selected layers plot...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # --- Data Preparation ---
    df_selected = df[df['layer_selection'] == "Selected Layers"].copy()

    # 1. Define the explicit order of all methods for plotting
    methods_order = [
        'Deterministic', 'Temp Sampling',  # Baselines
        'MC Dropout', 'Ensemble', 'SWAG',   # Family 1
        'MFVI', 'Full VB',                 # Family 2
        'Variational Temp'                 # Family 3
    ]
    baseline_methods = ['Deterministic', 'Temp Sampling']

    datasets = ["OBQA", "MMLU-Law", "MedMCQA-Med"]
    metrics = ["ACC", "NLL", "ECE", "MCE"]
    metric_labels = {
        "ACC": "ACC" + r"$\uparrow$", 
        "NLL": "NLL" + r"$\downarrow$", 
        "ECE": "ECE" + r"$\downarrow$", 
        "MCE": "MCE" + r"$\downarrow$"
    }
    
    palette = sns.color_palette("tab10", n_colors=len(methods_order))
    color_map = {method: color for method, color in zip(methods_order, palette)}

    fig, axes = plt.subplots(len(datasets), len(metrics), 
                             figsize=(22, 14), 
                            #  sharey='row',
                             gridspec_kw={'wspace': 0.15, 'hspace': 0.15})

    # fig.suptitle("ID Calibration Performance (Selected Layers) with Baselines", fontsize=24, y=0.98)

    for row_idx, dataset in enumerate(datasets):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            subplot_data = df_selected[(df_selected['dataset'] == dataset) & (df_selected['metric'] == metric)]
            
            # --- 2. Plot All Methods as Bars in the Predefined Order ---
            sns.barplot(
                data=subplot_data,
                x='method',
                y='mean_value',
                ax=ax,
                palette=color_map,
                edgecolor='black',
                order=methods_order
            )
            
            # --- 3. Add Error Bars, Skipping Baselines ---
            for i, method in enumerate(methods_order):
                # This is the new condition to skip error bars for baselines
                if method in baseline_methods:
                    continue
                
                data_point = subplot_data[subplot_data['method'] == method]
                if not data_point.empty:
                    mean = data_point['mean_value'].iloc[0]
                    std_dev = data_point['std_dev'].iloc[0]
                    ax.errorbar(x=i, y=mean, yerr=std_dev, fmt='none', c='black', capsize=3)

            # --- 4. Add All Vertical Separators ---
            # Separator between baselines and Bayesian methods (after index 1)
            ax.axvline(x=1.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
            # Separator after the first family (after index 4)
            ax.axvline(x=4.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
            # Separator after the second family (after index 6)
            ax.axvline(x=6.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)

            # --- Aesthetics ---
            if row_idx == 0:
                ax.set_title(metric_labels[metric], fontsize=18, pad=15)
            if col_idx == 0:
                ax.set_ylabel(dataset, fontsize=18, labelpad=15)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', alpha=0.4)

    # --- Legend Creation ---
    # --- Grouped Legend Construction ---
    legend_groups = [
        {
            "title": "Baseline",
            "methods": ['Deterministic', 'Temp Sampling']
        },
        {
            "title": "Weight-Space Bayes",
            "methods": ['MC Dropout', 'Ensemble', 'SWAG']
        },
        {
            "title": "Logit-space Bayes",
            "methods": ['MFVI', 'Full VB']
        },
        {
            "title": "Routing-space Bayes",
            "methods": ['Variational Temp']
        }
    ]

    legend_handles = []
    legend_titles = []
    for group in legend_groups:
        handles = [mpatches.Patch(color=color_map[method], label=method) for method in group["methods"]]
        legend_handles.append(handles)
        legend_titles.append(group["title"])

    # Place grouped legends horizontally, spaced across the bottom
    legend_y = 0.04
    group_xs = [0.215, 0.415, 0.62, 0.81]  # Adjust these values for horizontal spread
    for i, (handles, title) in enumerate(zip(legend_handles, legend_titles)):
        fig.legend(
            handles=handles,
            loc='lower center',
            ncol=len(handles),
            title=title,
            bbox_to_anchor=(group_xs[i], legend_y),
            fontsize=12,
            title_fontsize=14,
            frameon=False
        )
    # --- Draw a rectangle around all legends ---
    # Calculate the bounding box that covers all legend groups
    legend_box = mpatches.FancyBboxPatch(
        (0.14, 0.06),  # (x, y) in figure coordinates
        0.73,          # width
        0.01,          # height
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor='gray',
        facecolor='none',
        transform=fig.transFigure,
        zorder=10
    )
    fig.patches.append(legend_box)

    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Saved final plot to '{output_image_path}'")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize calibration results.")
    parser.add_argument("--input_json", type=str, default="fake_calibration_data.json", help="Path to the input JSON file with calibration data.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save the output plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Call the original plotting function ---
    # detailed_plot_path = os.path.join(args.output_dir, "calibration_results_grid_detailed.pdf")
    # plot_calibration_grid(args.input_json, detailed_plot_path)

    # --- Call the new plotting function ---
    selected_layers_plot_path = os.path.join(args.output_dir, "calibration_results_grid_selected_layers.pdf")
    plot_selected_layers_legend_only(args.input_json, selected_layers_plot_path)

if __name__ == "__main__":
    main()
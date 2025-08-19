import json
import random
import pandas as pd

def generate_fake_data(output_path="fake_calibration_data.json"):
    """
    Generates a structured JSON file with fake data for the calibration plot.
    """
    print("Generating fake data for visualization...")

    # --- Configuration ---
    datasets = ["MMLU-ID-HS", "MMLU-ID-SocialSci", "MMLU-ID-Humanities"]
    metrics = ["ACC", "NLL", "ECE", "MCE"]
    methods = [
        "Deterministic", "Temp Sampling", "SWAG", "MC Dropout",
        "Ensemble", "MFVI", "Full VB", "Variational Temp"
    ]
    layer_selections = ["Selected Layers", "All Layers"]

    # --- Plausible Value Ranges for Metrics ---
    # (Method, Layer Selection): (ACC_mean, NLL_mean, ECE_mean, MCE_mean)
    # We'll make Bayesian methods slightly better on average.
    plausible_means = {
        ("Deterministic", "All Layers"):     (0.70, 0.90, 0.15, 0.20),
        ("Deterministic", "Selected Layers"): (0.71, 0.88, 0.14, 0.19),
        ("Temp Sampling", "All Layers"):     (0.72, 0.85, 0.12, 0.18),
        ("Temp Sampling", "Selected Layers"): (0.73, 0.83, 0.11, 0.17),
        ("SWAG", "All Layers"):              (0.74, 0.80, 0.09, 0.15),
        ("SWAG", "Selected Layers"):         (0.75, 0.78, 0.08, 0.14),
        ("MC Dropout", "All Layers"):        (0.73, 0.82, 0.10, 0.16),
        ("MC Dropout", "Selected Layers"):    (0.74, 0.81, 0.09, 0.15),
        ("Ensemble", "All Layers"):          (0.76, 0.75, 0.07, 0.12),
        ("Ensemble", "Selected Layers"):     (0.77, 0.74, 0.06, 0.11),
        ("MFVI", "All Layers"):              (0.75, 0.77, 0.08, 0.13),
        ("MFVI", "Selected Layers"):         (0.76, 0.76, 0.07, 0.12),
        ("Full VB", "All Layers"):           (0.77, 0.73, 0.06, 0.11),
        ("Full VB", "Selected Layers"):      (0.78, 0.72, 0.05, 0.10),
        ("Variational Temp", "All Layers"): (0.74, 0.79, 0.09, 0.14),
        ("Variational Temp", "Selected Layers"): (0.75, 0.78, 0.08, 0.13),
    }

    all_results = []
    for dataset in datasets:
        for method in methods:
            for layer_selection in layer_selections:
                # Get the base mean values for this combination
                base_acc, base_nll, base_ece, base_mce = plausible_means[(method, layer_selection)]
                
                # Add some random noise to simulate variance between datasets
                acc = base_acc + random.uniform(-0.02, 0.02)
                nll = base_nll + random.uniform(-0.05, 0.05)
                ece = base_ece + random.uniform(-0.01, 0.01)
                mce = base_mce + random.uniform(-0.02, 0.02)
                
                # Create a record for each metric
                all_results.append({
                    "dataset": dataset, "metric": "ACC", "method": method,
                    "layer_selection": layer_selection, "mean_value": acc, "std_dev": acc * 0.05
                })
                all_results.append({
                    "dataset": dataset, "metric": "NLL", "method": method,
                    "layer_selection": layer_selection, "mean_value": nll, "std_dev": nll * 0.08
                })
                all_results.append({
                    "dataset": dataset, "metric": "ECE", "method": method,
                    "layer_selection": layer_selection, "mean_value": ece, "std_dev": ece * 0.15
                })
                all_results.append({
                    "dataset": dataset, "metric": "MCE", "method": method,
                    "layer_selection": layer_selection, "mean_value": mce, "std_dev": mce * 0.15
                })

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Successfully generated and saved fake data to '{output_path}'")

if __name__ == "__main__":
    generate_fake_data()
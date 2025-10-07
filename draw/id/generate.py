import json

def generate_fake_data(output_path="calibration_data.json"):
    """
    Generates a structured JSON file with specific fake data for the calibration plot.
    """
    print("Generating fake data for visualization...")

    # --- Configuration ---
    # Only one dataset now
    datasets = ["OBQA"]
    metrics = ["ACC", "NLL", "ECE", "MCE"]
    # Updated method names to match your legend
    methods = [
        "Deterministic", "Temp Sampling", "MCDR", "SWAGR",
        "DER", "MFVR", "FCVR", "VTSR"
    ]

    # --- New data extracted directly from your image ---
    # Structure: { method: { metric: (mean, std_dev) } }
    # For methods with no std_dev, it's set to 0.0
    specific_data = {
        "Deterministic": {
            "ACC": (0.746, 0.0), "NLL": (1.384, 0.0), "ECE": (0.252, 0.0), "MCE": (0.472, 0.0)
        },
        "Temp Sampling": {
            "ACC": (0.716, 0.005), "NLL": (0.773, 0.049), "ECE": (0.107, 0.009), "MCE": (0.201, 0.013)
        },
        "MCDR": {
            "ACC": (0.734, 0.002), "NLL": (0.650, 0.022), "ECE": (0.037, 0.028), "MCE": (0.298, 0.008)
        },
        "SWAGR": {
            "ACC": (0.736, 0.002), "NLL": (0.652, 0.030), "ECE": (0.041, 0.013), "MCE": (0.290, 0.007)
        },
        "DER": {
            "ACC": (0.738, 0.0), "NLL": (0.660, 0.0), "ECE": (0.071, 0.0), "MCE": (0.234, 0.0)
        },
        "MFVR": {
            "ACC": (0.742, 0.001), "NLL": (0.654, 0.019), "ECE": (0.026, 0.009), "MCE": (0.293, 0.004)
        },
        "FCVR": {
            "ACC": (0.740, 0.001), "NLL": (0.652, 0.021), "ECE": (0.015, 0.008), "MCE": (0.152, 0.004)
        },
        "VTSR": {
            "ACC": (0.736, 0.003), "NLL": (0.667, 0.025), "ECE": (0.052, 0.023), "MCE": (0.293, 0.014)
        }
    }

    all_results = []
    # Loop through the new data structure
    for dataset in datasets:
        for method in methods:
            for metric in metrics:
                # Get the mean and std_dev from our specific data
                mean_val, std_val = specific_data[method][metric]

                # Create a record for each metric
                all_results.append({
                    "dataset": dataset,
                    "metric": metric,
                    "method": method,
                    # Hardcode this to maintain compatibility with the plotting script
                    "layer_selection": "Selected Layers", 
                    "mean_value": mean_val,
                    "std_dev": std_val
                })

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Successfully generated and saved specific fake data to '{output_path}'")

if __name__ == "__main__":
    generate_fake_data()

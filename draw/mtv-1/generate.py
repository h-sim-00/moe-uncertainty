import pandas as pd
import numpy as np
import json

def generate_fake_data(output_path="jaccard_scores.json"):
    """
    Generates a toy dataset of Jaccard scores for visualization testing.
    """
    print("Generating fake data for visualization...")
    
    # --- Configuration ---
    num_layers = 32
    gammas = [0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05]
    num_tokens_per_layer = 500  # Number of data points per layer/gamma combination

    all_scores = []

    for gamma in gammas:
        for layer in range(num_layers):
            # Create a distribution that degrades with higher gamma and layer index
            # We use a Beta distribution, which is perfect for values between 0 and 1.
            # A higher 'a' parameter pushes the distribution towards 1 (high stability).
            # A higher 'b' parameter pushes it towards 0 (low stability).
            a = 10 - (gamma * 150) - (layer * 0.1)
            b = 1 + (gamma * 150) + (layer * 0.1)
            
            # Ensure parameters are positive
            a = max(a, 1)
            b = max(b, 1)

            # Generate random scores from the Beta distribution
            scores = np.random.beta(a, b, size=num_tokens_per_layer)
            
            for score in scores:
                all_scores.append({
                    "gamma": gamma,
                    "layer": layer,
                    "jaccard_score": score
                })

    # Convert to a JSON-compatible format (list of dicts)
    with open(output_path, 'w') as f:
        json.dump(all_scores, f, indent=4)
        
    print(f"Successfully generated and saved fake data to '{output_path}'")

if __name__ == "__main__":
    generate_fake_data()
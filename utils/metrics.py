import torch

def calculate_accuracy(preds, labels):
    """Calculates accuracy."""
    correct = (preds == labels).sum()
    return correct / len(labels)

def calculate_nll(probs, labels):
    """Calculates Negative Log-Likelihood (NLL)."""
    # Select the probability of the correct class
    correct_class_probs = probs[torch.arange(len(labels)), labels]
    # Prevent log(0) by clipping probabilities
    correct_class_probs = torch.clamp(correct_class_probs, min=1e-9)
    nll = -torch.log(correct_class_probs).mean()
    return nll

def calculate_ece_mce(probs, labels, n_bins=10):
    """Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            abs_diff = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * abs_diff
            mce = max(mce, abs_diff)

    return ece, mce

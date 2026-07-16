import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm

from .base import MoERouter # Import the base class
from utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll

class VariationalTemperatureRouter(MoERouter):
    """
    Implements a router with a learned, data-dependent temperature.
    """
    def __init__(self, config, existing_router):
        # Initialize the parent class, which creates self.layer
        super().__init__(config, existing_router)
        
        # Freeze the linear layer, as it represents the pre-trained MAP estimate
        for param in self.layer.parameters():
            param.requires_grad = False

        # Create a new, trainable network to predict the temperature(s)
        self.temperature_net = nn.Sequential(
            nn.Linear(self.input_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
        # Use Softplus to ensure the temperature is always positive
        self.softplus = nn.Softplus()

        # Captured on each forward for OoD-signal readout at eval time:
        #   last_temperature   -> Inf-Temp signal (raw T_phi, paper Eq. 31)
        #   last_scaled_logits -> Gate-Ent signal (entropy of softmax(l_det / T_phi))
        self.last_temperature = None
        self.last_scaled_logits = None

        # Diagnostic toggle (eval only). When True, inference uses DETERMINISTIC Top-K
        # (the K highest-prob experts) instead of the paper's stochastic Sample-K
        # (multinomial draw from softmax(l_det/T)). Lets us isolate whether an ID-accuracy
        # collapse comes from an inflated T + random sampling, or from a wiring bug: if
        # accuracy recovers with this ON, sampling under a runaway T is the culprit.
        self.deterministic_inference = False

    def forward(self, hidden_states, **kwargs):
        with torch.no_grad():
            logits = self.layer(hidden_states).float()

        temperature = self.softplus(self.temperature_net(hidden_states)) + 1e-6

        self.last_temperature = temperature

        scaled_logits = logits / temperature

        self.last_scaled_logits = scaled_logits
        
        # --- Conditional Logic for Training vs. Evaluation ---
        # `weight_logits` are the logits whose Top-K softmax becomes the gate weights.
        # They MUST depend on `temperature` so the reconstruction loss can push back on T.
        if self.training:
            # --- Algorithm 3 (paper App. C): Top-K over Gumbel-perturbed, T-scaled logits ---
            # Add Gumbel(0,1) noise to the RAW logits, THEN divide by T -> softmax((l_det + g)/T),
            # so the temperature modulates the exploration noise. Then keep the true Top-K experts
            # and softmax their T-scaled logits as soft gate weights.
            #
            # This replaces Listing 2's `gumbel_softmax(l_det/T, hard=True)` + `topk(one_hot)`,
            # which (a) injected fixed-scale noise T could not modulate [= softmax(l_det/T + g)]
            # and (b) selected 1 real expert + (K-1) arbitrary index-order residue experts. Under
            # (b) the chosen experts barely depended on T, so reconstruction gave T almost no
            # gradient and the unbounded `-log(T)` penalty ran T away (temperature explosion).
            # Selecting the true Top-K and weighting by their T-scaled logits restores that
            # counter-gradient. NOTE: this follows Algorithm 3, NOT the paper's own Listing 2.
            gumbels = -torch.empty_like(logits).exponential_().log()   # ~ Gumbel(0, 1)
            weight_logits = (logits + gumbels) / temperature
            top_k_indices = torch.topk(weight_logits, self.top_k, dim=1).indices
        elif self.deterministic_inference:
            # Deterministic Top-K: the K highest-prob experts, no sampling. Matches how
            # the MAP router selects, so accuracy here is the sampling-free ceiling.
            weight_logits = scaled_logits
            top_k_indices = torch.topk(scaled_logits.float(), self.top_k, dim=1).indices
        else:
            # Paper's Sample-K at inference: draw K experts without replacement from softmax(l_det/T).
            weight_logits = scaled_logits
            probabilities = torch.softmax(scaled_logits.float(), dim=1)
            top_k_indices = torch.multinomial(probabilities, self.top_k, replacement=False)

        gathered_logits = weight_logits.gather(1, top_k_indices.long())
        top_k_gates = torch.softmax(gathered_logits, dim=1).type_as(hidden_states)

        # --- The rest of the routing logic is now consistent ---
        batch_size = hidden_states.shape[0]
        zeros = torch.zeros((batch_size, self.num_experts), dtype=torch.long, device=logits.device)
        gates = zeros.scatter(1, top_k_indices.long(), 1)
        expert_size = gates.long().sum(0).tolist()
        num_selected_experts = top_k_indices.shape[1]
        top_k_experts = top_k_indices.flatten()
        _, index_sorted_experts = top_k_experts.sort(0)
        batch_index = index_sorted_experts.div(num_selected_experts, rounding_mode="trunc")
        
        flat_top_k_gates = top_k_gates.flatten()
        batch_gates = flat_top_k_gates[index_sorted_experts]
        
        return index_sorted_experts, batch_index, batch_gates, expert_size, scaled_logits
    
    def save_weights(self, path: str):
        """Saves the state_dict of only the trainable temperature network."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.temperature_net.state_dict(), path)
        print(f"Saved VTSR weights to {path}")

    def load_weights(self, path: str, device=None):
        """Loads the state_dict for the trainable temperature network."""
        state_dict = torch.load(path, map_location=device)
        self.temperature_net.load_state_dict(state_dict)
        print(f"Loaded VTSR weights from {path}")

def evaluate_vtsr(model, tokenizer, dataset, dataset_name, batch_size):
    """
    Orchestrates evaluation for the VariationalTemperatureRouter.
    Since the VTR is deterministic at inference, no MC sampling is needed.
    """
    print(f"--- Evaluating on {dataset_name} ---")
    model.eval()

    preds, probs, labels = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
    
    acc = calculate_accuracy(preds, labels)
    nll = calculate_nll(probs, labels)
    ece, mce = calculate_ece_mce(probs, labels)
    
    results = {
        'dataset': dataset_name, 'ACC': acc.item(), 'NLL': nll.item(),
        'ECE': ece.item(), 'MCE': mce.item(),
    }
    print(results)
    return results
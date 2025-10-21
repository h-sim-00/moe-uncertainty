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

        self.last_temperature = None

    def forward(self, hidden_states, **kwargs):
        with torch.no_grad():
            logits = self.layer(hidden_states).float()

        temperature = self.softplus(self.temperature_net(hidden_states)) + 1e-6

        self.last_temperature = temperature

        scaled_logits = logits / temperature
        
        # --- Conditional Logic for Training vs. Evaluation ---
        if self.training:
            # Use Gumbel-Softmax for a differentiable sample.
            # `hard=True` uses a one-hot vector in the forward pass but a soft,
            # differentiable approximation in the backward pass.
            gumbel_probs = F.gumbel_softmax(scaled_logits, tau=1.0, hard=True, dim=-1)
            top_k_indices = torch.topk(gumbel_probs, self.top_k, dim=1).indices
        else:
            probabilities = torch.softmax(scaled_logits.float(), dim=1)
            top_k_indices = torch.multinomial(probabilities, self.top_k, replacement=False)

        gathered_logits = scaled_logits.gather(1, top_k_indices.long())
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
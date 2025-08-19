import torch
from torch import nn
import os
from tqdm import tqdm

from .base import MoERouter # Import the base class
from ...utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll

class VariationalTemperatureRouter(MoERouter):
    """
    Implements a router with a learned, data-dependent temperature.
    This can operate in two modes:
    1. 'per_expert': Learns a unique temperature for each expert.
    2. 'shared': Learns a single temperature applied to all experts.
    """
    def __init__(self, config, existing_router, temperature_mode='per_expert'):
        # Initialize the parent class, which creates self.layer
        super().__init__(config, existing_router)
        
        # Freeze the linear layer, as it represents the pre-trained MAP estimate
        for param in self.layer.parameters():
            param.requires_grad = False

        self.temperature_mode = temperature_mode
        
        # The output dimension of the network depends on the mode
        if self.temperature_mode == 'per_expert':
            output_dim = self.num_experts
        elif self.temperature_mode == 'shared':
            output_dim = 1
        else:
            raise ValueError(f"Invalid temperature_mode: {temperature_mode}. Choose 'per_expert' or 'shared'.")

        # Create a new, trainable network to predict the temperature(s)
        self.temperature_net = nn.Sequential(
            nn.Linear(self.input_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, output_dim)
        )
        
        # Use Softplus to ensure the temperature is always positive
        self.softplus = nn.Softplus()

    def forward(self, hidden_states, **kwargs):
        """
        Overrides the forward pass to apply the learned temperature.
        """
        # 1. Get the deterministic logits from the frozen base layer
        with torch.no_grad():
            logits = self.layer(hidden_states).float()

        # 2. Predict the temperature(s) from the trainable network
        raw_temp = self.temperature_net(hidden_states)
        
        # 3. Apply Softplus and add epsilon for stability
        temperatures = self.softplus(raw_temp) + 1e-6
        
        # If in shared mode, temperatures will have shape [batch, 1],
        # so it will broadcast correctly during division.
        
        # 4. Apply the learned temperature to the logits
        scaled_logits = logits / temperatures
        
        # 5. The rest of the routing logic uses these new scaled_logits
        top_k_logits, top_k_indices = scaled_logits.topk(self.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)
        
        batch_size = hidden_states.shape[0]
        zeros = torch.zeros((batch_size, self.num_experts), dtype=torch.long, device=logits.device)
        gates = zeros.scatter(1, top_k_indices.long(), 1)
        expert_size = gates.long().sum(0).tolist()
        num_selected_experts = top_k_indices.shape[1]
        top_k_experts = top_k_indices.flatten()
        _, index_sorted_experts = top_k_experts.sort(0)
        batch_index = index_sorted_experts.div(num_selected_experts, rounding_mode="trunc")
        top_k_gates = top_k_gates.flatten()
        batch_gates = top_k_gates[index_sorted_experts]
        
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
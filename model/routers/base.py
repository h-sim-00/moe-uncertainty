# model/routers/base.py

import torch
from torch import nn
import os

class MoERouter(nn.Module):
    """
    A modular and reusable base class for the MoE routing mechanism, designed
    to be a drop-in replacement for the original router and a foundation for
    all subsequent Bayesian router implementations.
    """
    def __init__(self, config, existing_router=None):
        """
        Initializes the MoERouter.

        Args:
            config (object): A configuration object with attributes like hidden_size,
                             num_local_experts, and num_experts_per_tok.
            existing_router (nn.Module, optional): An existing router module (like
                                                   GraniteMoeTopKGating) from which to
                                                   copy initial weights. Defaults to None.
        """
        super().__init__()
        self.num_experts = config.num_local_experts
        self.input_size = config.hidden_size
        self.top_k = config.num_experts_per_tok
        
        self.layer = nn.Linear(self.input_size, self.num_experts, bias=False)

        if existing_router is not None:
            print("Initializing MoERouter weights from an existing router.")
            self.layer.load_state_dict(existing_router.layer.state_dict())

    def forward(self, hidden_states, mode="top_k", temp=1.0):
        """
        The forward pass is identical to the original GraniteMoeTopKGating logic
        to ensure compatibility.
        """
        logits = self.layer(hidden_states).float()
        batch_size = hidden_states.shape[0]

        if mode == "top_k":
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
            top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)

        # elif mode == "mc_dropout_original_logits":
        #     # For MC Dropout, the dropout layer must be active during inference.
        #     # Using original routing logits now
        #     self.dropout.train()
        #     stochastic_hidden_states = self.dropout(hidden_states)
        #     stochastic_logits = self.layer(stochastic_hidden_states).float()
        #     top_k_logits, top_k_indices = stochastic_logits.topk(self.top_k, dim=1)
        #     top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)
        
        # elif mode == "mc_dropout_stochastic_logits":
        #     self.dropout.train()
        #     stochastic_hidden_states = self.dropout(hidden_states)
        #     stochastic_logits = self.layer(stochastic_hidden_states).float()
        #     top_k_logits, top_k_indices = stochastic_logits.topk(self.top_k, dim=1)
        #     top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)
        #     logits = stochastic_logits  # Use stochastic logits for logging

        elif mode == "random_k":
            top_k_indices = torch.stack([
                torch.randperm(self.num_experts, device=logits.device)[:self.top_k]
                for _ in range(batch_size)
            ])
            top_k_gates = torch.full((batch_size, self.top_k), fill_value=1.0 / self.top_k, device=logits.device, dtype=hidden_states.dtype)

        elif mode == "fixed_k":
            top_k_indices = torch.arange(self.top_k, device=logits.device).unsqueeze(0).expand(batch_size, -1)
            top_k_gates = torch.full((batch_size, self.top_k), fill_value=1.0 / self.top_k, device=logits.device, dtype=hidden_states.dtype)

        elif mode == "all_weighted":
            top_k_indices = torch.arange(self.num_experts, device=logits.device).unsqueeze(0).expand(batch_size, -1)
            top_k_gates = torch.softmax(logits, dim=1).type_as(hidden_states)

        elif mode == "sample_k":
            scaled_logits = logits / temp
            probabilities = torch.softmax(scaled_logits.float(), dim=1)
            top_k_indices = torch.multinomial(probabilities, self.top_k, replacement=False)
            gathered_logits = logits.gather(1, top_k_indices.long()) 
            top_k_gates = torch.softmax(gathered_logits, dim=1).type_as(hidden_states)

        else:
            raise ValueError(f"Invalid routing mode: {mode}.")

        
        # This part remains the same for all k-expert modes
        zeros = torch.zeros((batch_size, self.num_experts), dtype=torch.long, device=logits.device)
        gates = zeros.scatter(1, top_k_indices.long(), 1)
        expert_size = gates.long().sum(0).tolist()
        num_selected_experts = top_k_indices.shape[1]
        top_k_experts = top_k_indices.flatten()
        _, index_sorted_experts = top_k_experts.sort(0)
        batch_index = index_sorted_experts.div(num_selected_experts, rounding_mode="trunc")
        top_k_gates = top_k_gates.flatten()
        batch_gates = top_k_gates[index_sorted_experts]
        
        return index_sorted_experts, batch_index, batch_gates, expert_size, logits

    def save_weights(self, path: str):
        """Saves the router's state_dict to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Saved router weights to {path}")

    def load_weights(self, path: str, device=None):
        """Loads the router's state_dict from a file."""
        if device is None:
            device = next(self.parameters()).device
        
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        print(f"Loaded router weights from {path}")
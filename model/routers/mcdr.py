# model/routers/mcdropout.py

import torch
from torch import nn
import os
from tqdm import tqdm

from .base import MoERouter # Import the base class
from ...utils import get_model_predictions, calculate_accuracy, calculate_ece_mce, calculate_nll

class MCDropoutRouter(MoERouter):
    """
    An MoE router that implements MC Dropout.
    - Inherits from the base MoERouter.
    - Keeps dropout active in both train and eval modes.
    - Performs Monte Carlo sampling internally during evaluation.
    """
    def __init__(self, config, existing_router=None, dropout_rate=0.1):
        # Initialize the parent class (handles self.layer and weight copying)
        super().__init__(config, existing_router)
        
        self.dropout = nn.Dropout(dropout_rate)
        # This will be set externally before evaluation
        self.num_mc_samples = 1 

    def train(self, mode: bool = True):
        """
        Overrides the default train method to ensure dropout is always active.
        This is the key to persistent dropout.
        """
        # Set the mode for the parent class (e.g., for the nn.Linear layer)
        super().train(mode)
        # Force the dropout layer to always be in training mode
        self.dropout.train()
        return self

    def forward(self, hidden_states, **kwargs):
        """
        Implements a conditional forward pass.
        - During training (self.training=True): a single stochastic pass.
        - During evaluation (self.training=False): internal Monte Carlo sampling.
        """
        # If the model is in .train() mode, perform a single stochastic forward pass.
        if self.training:
            stochastic_hidden_states = self.dropout(hidden_states)
            logits = self.layer(stochastic_hidden_states).float()
        
        # If the model is in .eval() mode, perform internal Monte Carlo sampling.
        else:
            all_logits = []
            for _ in range(self.num_mc_samples):
                # Our overridden train() method ensures self.dropout is always active.
                stochastic_hidden_states = self.dropout(hidden_states)
                logits_sample = self.layer(stochastic_hidden_states).float()
                all_logits.append(logits_sample)
            
            # Average the logits across all samples to get a robust estimate
            logits = torch.stack(all_logits).mean(dim=0)
            
        # The rest of the routing logic uses the calculated logits (either single or averaged)
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
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
        
        return index_sorted_experts, batch_index, batch_gates, expert_size, logits

def evaluate_mcdropout_router(model, tokenizer, dataset, dataset_name, num_samples, batch_size):
    """
    Orchestrates evaluation for the MCDropoutRouter.
    """
    print(f"--- Evaluating on {dataset_name} with {num_samples} MC samples ---")
    
    # Set the number of MC samples for the forward pass
    causal_model = model.base_model.model.model
    for layer in causal_model.layers:
        if isinstance(layer.block_sparse_moe.router, MCDropoutRouter):
            layer.block_sparse_moe.router.num_mc_samples = num_samples

    model.eval() # This will trigger the MC sampling logic in the router's forward pass

    # We now only need one pass of get_model_predictions, as the sampling is internal
    preds, probs, labels = get_model_predictions(model, tokenizer, dataset, batch_size=batch_size)
    
    # Calculate metrics
    acc = calculate_accuracy(preds, labels)
    nll = calculate_nll(probs, labels)
    ece, mce = calculate_ece_mce(probs, labels)
    
    results = {
        'dataset': dataset_name, 'ACC': acc.item(), 'NLL': nll.item(),
        'ECE': ece.item(), 'MCE': mce.item(),
    }
    print(results)
    return results
# In place of GraniteMoeTopKGating in model.granitemoe
from torch import nn
import torch

class GraniteMoeMCDropoutRouter(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int, dropout_rate: float = 0.1):
        """
        Initialize the gating mechanism.

        Args:
            input_size (`int`): Size of the input.
            num_experts (`int`): Number of experts.
            top_k (`int`): Number of top experts to select in "top_k" mode.
            fixed_k (`int`): Fixed number of experts to use in "fixed_k" mode.
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k

        self.layer = nn.Linear(input_size, num_experts, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, mode="top_k", temp=1.0):
        """
        Forward method for the gating mechanism.

        Args:
            hidden_states (`torch.Tensor`): Input hidden states of shape [batch_size * seq_len, input_size].
            mode (`str`): Routing mode. One of "top_k", "sample_k", "mc_dropout", etc.
            temp (`float`): Temperature for "sample_k" mode.

        Returns:
            index_sorted_experts (`torch.Tensor`): Indices of selected experts, sorted for efficient processing.
            batch_index (`torch.Tensor`): Batch index for grouped inputs.
            batch_gates (`torch.Tensor`): Gate values for grouped inputs.
            expert_size (`List[int]`): Number of tokens assigned to each expert.
            logits (`torch.Tensor`): Original (pre-dropout) logits from the gating layer.
        """
        # Note: The 'logits' returned will be based on the original hidden_states for consistency in logging.
        # The stochasticity is only applied internally for the mc_dropout mode.
        logits = self.layer(hidden_states).float()
        batch_size = hidden_states.shape[0]

        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)

        # This part remains the same for all k-expert modes
        zeros = torch.zeros(
            (batch_size, self.num_experts), dtype=torch.long, device=logits.device
        )
        gates = zeros.scatter(1, top_k_indices.long(), 1)
        expert_size = gates.long().sum(0).tolist()

        num_selected_experts = top_k_indices.shape[1]
        top_k_experts = top_k_indices.flatten()
        _, index_sorted_experts = top_k_experts.sort(0)
        
        batch_index = index_sorted_experts.div(num_selected_experts, rounding_mode="trunc")
        
        top_k_gates = top_k_gates.flatten()
        batch_gates = top_k_gates[index_sorted_experts]

        return index_sorted_experts, batch_index, batch_gates, expert_size, logits
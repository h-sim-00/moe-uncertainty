import torch
from torch import nn
import torch.nn.functional as F
from model.routers.base import MoERouter

class BayesianQwenMoeBlock(nn.Module):
    """
    A 'Lego piece' replacement for Qwen2MoeSparseMoeBlock.

    This module mimics the external behavior of the original Qwen MoE block but
    uses our modular Bayesian routers internally, without changing their API.
    """
    def __init__(self, config, original_qwen_block, RouterClass, **router_kwargs):
        super().__init__()
        print(f"Instantiating BayesianQwenMoeBlock with {RouterClass.__name__}")

        # Unchanged components
        self.shared_expert = original_qwen_block.shared_expert
        self.shared_expert_gate = original_qwen_block.shared_expert_gate
        self.experts = original_qwen_block.experts

        # Reinitialisation of the detereministic router
        class DummyRouterWrapper:
            def __init__(self, gate_layer):
                self.layer = gate_layer
        dummy_router_for_init = DummyRouterWrapper(original_qwen_block.gate)
        self.router = RouterClass(config, existing_router=dummy_router_for_init, **router_kwargs)

        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, _ = hidden_states.shape
        # Flatten the input: tokens as units
        flat_hidden_states = hidden_states.view(-1, self.hidden_dim)

        # --- Step A: Run our Bayesian Router ---
        _, _, _, _, router_logits = self.router(flat_hidden_states)
        
        # Access the indices stored by the router during its forward pass
        selected_experts = self.router.last_top_k_indices

        # --- Step B: Dispatch to MoE Experts (Qwen-style) ---
        moe_output = torch.zeros_like(flat_hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        routing_weights, _ = torch.topk(F.softmax(router_logits, dim=1, dtype=torch.float), self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue

            current_state = flat_hidden_states[None, top_x].reshape(-1, self.hidden_dim)
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
            moe_output.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # --- Step C: Compute and Add the Shared Expert Output ---
        shared_expert_output = self.shared_expert(flat_hidden_states)
        shared_expert_gate_output = self.shared_expert_gate(flat_hidden_states)
        shared_expert_output = F.sigmoid(shared_expert_gate_output) * shared_expert_output
        
        final_hidden_states = moe_output + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.hidden_dim)

        # --- Step D: Match the Return Signature ---
        return final_hidden_states, router_logits


# --- SURGEON 1: The Outer Surgery ---
def swap_qwen_moe_blocks(model):
    """
    Performs the initial, full architectural swap.
    Replaces all native Qwen MoE blocks with our BayesianQwenMoeBlock container,
    always initializing them with a base, deterministic MoERouter.
    """

    # Freeze all original model parameters
    for param in model.parameters():
        param.requires_grad = False  

    # Adapting all routing layers, load old parameters, unfreeze new parameters
    print("--- Starting Outer Surgery: Adapting Qwen architecture ---")
    causal_model = model.base_model.model
    moe_model = causal_model.model
    config = model.config

    for layer in moe_model.layers:
        original_qwen_block = layer.mlp
        new_container_block = BayesianQwenMoeBlock(config, original_qwen_block, MoERouter)
        new_container_block.to(model.device)
        layer.mlp = new_container_block
        router_linear = new_container_block.router
        for param in router_linear.parameters():
            param.requires_grad = True 
        
    print("--- Outer Surgery complete ---")
    print("--- Ready for deterministic router finetuning ---")
    return model


# --- SURGEON 2: The Inner Surgery ---
def replace_qwen_internal_routers(model, RouterClass, swap_layers=[], **router_kwargs):
    """
    Performs a targeted, inner swap on an already adapted model.
    Reaches inside the BayesianQwenMoeBlock container at specified layers
    and replaces the deterministic router with a specific Bayesian one.
    """
    print(f"--- Starting Inner Surgery: Swapping internal routers to {RouterClass.__name__} ---")
    causal_model = model.base_model.model
    moe_model = causal_model.model
    config = model.config

    for layer_idx in swap_layers:
        print(f"  - Swapping router in layer {layer_idx}")
        # Navigate to the container block
        container_block = moe_model.layers[layer_idx].mlp
        
        # The "existing_router" is now the deterministic router already inside our container
        existing_det_router = container_block.router
        
        # Instantiate the new Bayesian router, inheriting weights from the deterministic one
        new_bayesian_router = RouterClass(config, existing_router=existing_det_router, **router_kwargs)
        
        # Perform the inner swap
        container_block.router = new_bayesian_router.to(model.device)

    print("--- Inner Surgery complete ---")
    return model
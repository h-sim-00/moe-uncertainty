import torch, os
from torch import nn
import torch.nn.functional as F

from model.routers.base import MoERouter
from model.routers.mcdr import MCDropoutRouter
from model.routers.mfvr import MeanFieldVariationalRouter
from model.routers.fcvr import FullCovarianceVariationalRouter
from model.routers.vtsr import VariationalTemperatureRouter

ROUTER_CONFIG = {
    "mcdr": {
        "class": MCDropoutRouter,
        "get_kwargs": lambda args: {"dropout_rate": args.dropout_rate},
        "trainable_attrs": None
    },
    "mfvr": {
        "class": MeanFieldVariationalRouter,
        "get_kwargs": lambda args: {},
        "trainable_attrs": ["mean_residual_net", "log_var_net"]
    },
    "fcvr": {
        "class": FullCovarianceVariationalRouter,
        "get_kwargs": lambda args: {},
        "trainable_attrs": ["backbone", "mean_head", "cholesky_head"]
    },
    "vtsr": {
        "class": VariationalTemperatureRouter,
        "get_kwargs": lambda args: {},
        "trainable_attrs": ["temperature_net"]
    }
}

class BayesianQwenMoeBlock(nn.Module):
    """
    A 'Lego piece' replacement for Qwen2MoeSparseMoeBlock.

    This module mimics the external behavior of the original Qwen MoE block but
    uses our modular Bayesian routers internally, without changing their API.
    """
    def __init__(self, config, original_qwen_block, RouterClass, **router_kwargs):
        super().__init__()
        self.shared_expert = original_qwen_block.shared_expert
        self.shared_expert_gate = original_qwen_block.shared_expert_gate
        self.experts = original_qwen_block.experts

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
        flat_hidden_states = hidden_states.view(-1, self.hidden_dim)
        _, _, _, _, router_logits = self.router(flat_hidden_states)
        selected_experts = self.router.last_top_k_indices
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

        shared_expert_output = self.shared_expert(flat_hidden_states)
        shared_expert_gate_output = self.shared_expert_gate(flat_hidden_states)
        shared_expert_output = F.sigmoid(shared_expert_gate_output) * shared_expert_output
        
        final_hidden_states = moe_output + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.hidden_dim)

        return final_hidden_states, router_logits

# ====================================================
# MAP (Deterministic) Router Functions
# ====================================================

# (0) Outer Surgery: Helper Function
def _perform_qwen_outer_surgery(model):
    """
    A private helper to perform the one-time architectural swap.
    This is the core of adapting Qwen to our framework.
    """
    causal_model = model.base_model.model.model
    config = model.config
    for layer in causal_model.layers:
        # If it's already our container, skip it.
        if isinstance(layer.mlp, BayesianQwenMoeBlock):
            continue
        original_qwen_block = layer.mlp
        new_container_block = BayesianQwenMoeBlock(config, original_qwen_block, MoERouter)
        layer.mlp = new_container_block.to(model.device)
    return model

# (1) Initilisation: For MAP det router finetuning
def swap_qwen_moe_blocks(model):
    """
    Prepares a Qwen model for MAP router fine-tuning.
    1. Performs architectural swap.
    2. Freezes all params.
    3. Unfreezes only the new base routers.
    """
    model = _perform_qwen_outer_surgery(model)
    print("--- Preparing model for MAP router tuning ---")
    for param in model.parameters():
        param.requires_grad = False
    causal_model = model.base_model.model.model
    for layer in causal_model.layers:
        for param in layer.mlp.router.parameters():
            param.requires_grad = True
    return model

# (2) Saving: save finetuned map weights
def save_qwen_map_routers(model, args):
    """Saves the fine-tuned MAP router weights for a Qwen model."""
    print("--- Saving Qwen MAP router weights ---")
    output_dir = "./router_weights/base"
    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    causal_model = model.base_model.model.model
    for i, layer in enumerate(causal_model.layers):
        save_path = os.path.join(output_dir, run_name, f"layer_{i}_weights.pt")
        # Note the updated path: layer.mlp.router
        layer.mlp.router.save_weights(save_path)

# (3) Loading: load finetuned map weights
def load_qwen_map_routers(model, args):
    """Loads pre-trained MAP router weights into an adapted Qwen model."""
    model = _perform_qwen_outer_surgery(model)
    print("--- Loading Qwen MAP routers ---")
    map_run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    map_weights_dir = f"./router_weights/base/{map_run_name}"
    causal_model = model.base_model.model.model
    for i, layer in enumerate(causal_model.layers):
        map_weights_path = os.path.join(map_weights_dir, f"layer_{i}_weights.pt")
        if os.path.exists(map_weights_path):
             layer.mlp.router.load_weights(map_weights_path, device=model.device)
    return model

# ====================================================
# Generic Bayesian Router Functions (New Elegant API)
# ====================================================

# (4) Preparing for Bayeisan Tuning
def prepare_qwen_bayesian_routers(model, method, args):
    """
    A generic function to prepare a Qwen model for Bayesian router training.
    """
    # 1. Ensure the base architecture is correct and MAP weights are loaded
    model = load_qwen_map_routers(model, args)
    
    # 2. Perform the "Inner Surgery" to swap in the Bayesian router
    if method not in ROUTER_CONFIG:
        raise ValueError(f"Unknown Bayesian method: {method}.")
    config = ROUTER_CONFIG[method]
    RouterClass = config["class"]
    router_kwargs = config["get_kwargs"](args)
    trainable_attrs = config["trainable_attrs"]
    
    print(f"--- Preparing Qwen for {method.upper()} router tuning ---")
    causal_model = model.base_model.model.model
    for layer_idx in args.swap_layers:
        container_block = causal_model.layers[layer_idx].mlp
        existing_det_router = container_block.router
        new_bayesian_router = RouterClass(config=model.config, existing_router=existing_det_router, **router_kwargs)
        # Load weights if specified
        # (Weight loading logic can be added here if needed)
        container_block.router = new_bayesian_router.to(model.device)

    # 3. Freeze & Unfreeze
    for param in model.parameters():
        param.requires_grad = False

    print(f"Unfreezing routers in layers: {args.train_layers}")
    for layer_idx in args.train_layers:
        router = causal_model.layers[layer_idx].mlp.router
        if trainable_attrs is None:
            for param in router.parameters():
                param.requires_grad = True
        else:
            for attr_name in trainable_attrs:
                submodule = getattr(router, attr_name)
                for param in submodule.parameters():
                    param.requires_grad = True
    return model

# (5) Save Bayesian Parameters
def save_qwen_bayesian_routers(model, method, args):
    """A generic function to save the weights of trained Bayesian routers from a Qwen model."""
    print(f"--- Saving Qwen {method.upper()} router weights ---")
    output_root_dir = f"./router_weights/{method}"
    if method == 'vtsr':
        output_root_dir += f"_{args.temperature_mode}"
    run_name = f"{method}-{args.model_shortcode}-{args.dataset_shortcode}"
    save_dir = os.path.join(output_root_dir, run_name)
    causal_model = model.base_model.model.model
    for layer_idx in args.swap_layers:
        save_path = os.path.join(save_dir, f"layer_{layer_idx}_weights.pt")
        causal_model.layers[layer_idx].mlp.router.save_weights(save_path)

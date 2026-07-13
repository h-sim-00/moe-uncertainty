import os
import torch
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


# ====================================================
# Low-Level Components (The "Lego Pieces")
# ====================================================

class BayesianDeepseekMoeBlock(nn.Module):
    """
    The 'Lego piece' replacement for DeepseekMoE.
    It acts as a container for our standard MoERouter subclasses and perfectly
    replicates the original DeepseekMoE forward pass logic.
    """
    def __init__(self, config, original_deepseek_block, RouterClass, **router_kwargs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        # 1. Steal the experts and shared experts from the original block
        self.experts = original_deepseek_block.experts
        if hasattr(original_deepseek_block, 'shared_experts'):
            self.shared_experts = original_deepseek_block.shared_experts
        else:
            self.shared_experts = None

        # 2. Create a dummy router to initialize our Bayesian router from the original gate
        class DummyRouterWrapper:
            def __init__(self, gate_module):
                self.layer = nn.Linear(gate_module.gating_dim, gate_module.n_routed_experts, bias=False)
                self.layer.weight.data = gate_module.weight.data.clone()

        dummy_router_for_init = DummyRouterWrapper(original_deepseek_block.gate)
        self.router = RouterClass(config, existing_router=dummy_router_for_init, **router_kwargs)

        self.last_router_logits = None

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # --- Step A: Run our Bayesian Router ---
        _, _, _, _, router_logits = self.router(flat_hidden_states)
        
        self.last_router_logits = router_logits

        # --- Step B: Re-create the outputs needed by the original DeepseekMoE forward pass ---
        topk_idx = self.router.last_top_k_indices
        
        # Re-calculate topk_weight from our router's outputs
        # This ensures perfect compatibility with the downstream logic.
        scores = router_logits.softmax(dim=-1)
        topk_weight = torch.gather(scores, 1, topk_idx)
        topk_weight /= topk_weight.sum(dim=-1, keepdim=True) # Normalize

        # --- Step C: Execute the original DeepseekMoE forward logic ---
        # This logic is copied directly from the source to ensure identical behavior.
        y = torch.empty((flat_hidden_states.shape[0], self.num_experts_per_tok, hidden_states.shape[-1]),
                        dtype=hidden_states.dtype, device=hidden_states.device)
        
        for i, expert in enumerate(self.experts):
            token_indices, expert_indices = torch.where(topk_idx == i)
            if token_indices.numel() > 0:
                y[token_indices, expert_indices, :] = expert(flat_hidden_states[token_indices, :])
        
        y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(*orig_shape)
        
        # Add shared expert output if it exists
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
            
        return y 

# ====================================================
# High-Level API (The Workflow Manager)
# ====================================================

def _perform_deepseek_outer_surgery(model):
    """
    A private helper to perform the one-time architectural swap for Deepseek.
    It intelligently skips the first layer which is not an MoE layer.
    """
    # This check prevents running the surgery more than once
    if hasattr(model, '_deepseek_surgery_performed'):
        return model

    print("--- Performing Deepseek Outer Surgery: Adapting architecture ---")
    causal_model = model.base_model.model.model
    config = model.config

    # Iterate and only swap layers that contain a DeepseekMoE block
    for i, layer in enumerate(causal_model.layers):
        # The original DeepseekMoE class name might vary slightly,
        # so we check for the presence of a 'gate' attribute as a robust indicator.
        if hasattr(layer.mlp, 'gate'):
            original_moe_block = layer.mlp
            new_container_block = BayesianDeepseekMoeBlock(config, original_moe_block, MoERouter)
            layer.mlp = new_container_block.to(model.device)
    
    model._deepseek_surgery_performed = True
    print("--- Outer Surgery complete ---")
    return model

def swap_deepseek_moe_blocks(model):
    """
    Prepares a Deepseek model for MAP router fine-tuning.
    """
    model = _perform_deepseek_outer_surgery(model)
    print("--- Preparing model for MAP router tuning ---")
    for param in model.parameters():
        param.requires_grad = False
    causal_model = model.base_model.model.model
    for layer in causal_model.layers:
        if isinstance(layer.mlp, BayesianDeepseekMoeBlock):
            for param in layer.mlp.router.parameters():
                param.requires_grad = True
    return model

def save_deepseek_map_routers(model, args):
    """Saves the fine-tuned MAP router weights for a Deepseek model."""
    print("--- Saving Deepseek MAP router weights ---")
    output_dir = "./router_weights/base"
    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    causal_model = model.base_model.model.model
    for i, layer in enumerate(causal_model.layers):
        if isinstance(layer.mlp, BayesianDeepseekMoeBlock):
            save_path = os.path.join(output_dir, run_name, f"layer_{i}_weights.pt")
            layer.mlp.router.save_weights(save_path)

def load_deepseek_map_routers(model, args):
    """Loads pre-trained MAP router weights into an adapted Deepseek model."""
    model = _perform_deepseek_outer_surgery(model)
    print("--- Loading Deepseek MAP routers ---")
    map_run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    map_weights_dir = f"./router_weights/base/{map_run_name}"
    causal_model = model.base_model.model.model
    for i, layer in enumerate(causal_model.layers):
        if isinstance(layer.mlp, BayesianDeepseekMoeBlock):
            map_weights_path = os.path.join(map_weights_dir, f"layer_{i}_weights.pt")
            if os.path.exists(map_weights_path):
                 layer.mlp.router.load_weights(map_weights_path, device=model.device)
    return model

def prepare_deepseek_bayesian_routers(model, method, args):
    """A generic function to prepare a Deepseek model for Bayesian router training."""
    model = load_deepseek_map_routers(model, args)
    if method not in ROUTER_CONFIG:
        raise ValueError(f"Unknown Bayesian method: {method}.")
    
    config = ROUTER_CONFIG[method]
    RouterClass, router_kwargs, trainable_attrs = config["class"], config["get_kwargs"](args), config["trainable_attrs"]
    
    print(f"--- Preparing Deepseek for {method.upper()} router tuning ---")
    causal_model = model.base_model.model.model
    for layer_idx in args.swap_layers:
        if isinstance(causal_model.layers[layer_idx].mlp, BayesianDeepseekMoeBlock):
            container_block = causal_model.layers[layer_idx].mlp
            existing_det_router = container_block.router
            new_bayesian_router = RouterClass(config=model.config, existing_router=existing_det_router, **router_kwargs)
            container_block.router = new_bayesian_router.to(model.device)

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
                for param in getattr(router, attr_name).parameters():
                    param.requires_grad = True
    return model

def save_deepseek_bayesian_routers(model, method, args):
    """A generic function to save the weights of trained Bayesian routers from a Deepseek model."""
    print(f"--- Saving Deepseek {method.upper()} router weights ---")
    output_root_dir = f"./router_weights/{method}"
    if method == 'vtsr':
        output_root_dir += f"_{args.temperature_mode}"
    run_name = f"{method}-{args.model_shortcode}-{args.dataset_shortcode}"
    save_dir = os.path.join(output_root_dir, run_name)
    causal_model = model.base_model.model.model
    for layer_idx in args.swap_layers:
        if isinstance(causal_model.layers[layer_idx].mlp, BayesianDeepseekMoeBlock):
            save_path = os.path.join(save_dir, f"layer_{layer_idx}_weights.pt")
            causal_model.layers[layer_idx].mlp.router.save_weights(save_path)
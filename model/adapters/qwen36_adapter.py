"""Qwen3.6-35B-A3B (transformers>=5 `qwen3_5_moe`) adapter -- branch OBQA-qwen.

Mirrors model/adapters/granite_adapter.py function-for-function (same
router_weights/ path scheme, same map_suffix / run_suffix handling, same
FileNotFoundError semantics) for a model whose MoE block lives at `layer.mlp`:

    Qwen3_5MoeSparseMoeBlock
        .gate               Qwen3_5MoeTopKRouter  (weight: nn.Parameter [E, H]; NOT an nn.Linear)
        .experts            Qwen3_5MoeExperts     (fused gate_up_proj [E, 2I, H], down_proj [E, H, I])
        .shared_expert      Qwen3_5MoeMLP
        .shared_expert_gate nn.Linear(H, 1)

Granite needs no structural surgery because its `GraniteMoeMoE` already calls a
router that returns the 5-tuple our MoERouter / FCVR classes produce. Here the
whole block is replaced by `BayesianQwen35MoeBlock`, a container that keeps the
experts / shared expert and delegates routing to one of OUR router classes under
the attribute name `.router` (so `model.adapters.moe_router(layer)` finds it).

Two things differ from the (stale, Qwen1.5) qwen_adapter.py and are deliberate:
  * routing weights are derived from the `logits` the router RETURNS (for FCVR
    these are the MC-averaged log-mean-probs), never from a side attribute;
  * the routers are fp32 modules copied from the bf16 gate, so the container
    feeds them `x.float()` -- Cholesky / MVN sampling / KL stay fp32 exactly as
    on Granite (whose whole pipeline ran in fp32).
Top-k-then-softmax over the selected logits is algebraically identical to
Qwen's softmax-all -> top-k -> renormalise, so the MAP container reproduces the
native block up to bf16-vs-fp32 gate rounding (checked by smoke-qwen36-model.py).
"""
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.routers.base import MoERouter
from model.adapters.granite_adapter import ROUTER_CONFIG   # same method registry


# ====================================================
# Config / gate shims
# ====================================================
def router_config_for(config):
    """The router classes read `num_local_experts` (Granite's name); Qwen calls
    it `num_experts`. A tiny namespace avoids touching model/routers/*.py."""
    if hasattr(config, "num_local_experts"):
        return config
    return SimpleNamespace(
        hidden_size=config.hidden_size,
        num_local_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
    )


def seed_linear_from_gate(gate):
    """MoERouter / FCVR copy `existing_router.layer.state_dict()` into an
    nn.Linear(H, E, bias=False). Qwen's gate holds the same [E, H] matrix as a
    bare Parameter, so wrap a Linear around a (fp32) copy of it."""
    weight = gate.weight
    lin = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        lin.weight.copy_(weight.detach().float())
    return SimpleNamespace(layer=lin)


def _decoder(model):
    return model.base_model.model.model


def _layer_device(layer):
    return next(layer.parameters()).device


# ====================================================
# Container block
# ====================================================
class BayesianQwen35MoeBlock(nn.Module):
    """Drop-in replacement for Qwen3_5MoeSparseMoeBlock with a pluggable router."""

    def __init__(self, config, original_block, RouterClass=MoERouter, existing_router=None, **router_kwargs):
        super().__init__()
        self.experts = original_block.experts
        self.shared_expert = original_block.shared_expert
        self.shared_expert_gate = original_block.shared_expert_gate
        self.hidden_dim = config.hidden_size
        self.top_k = config.num_experts_per_tok
        if existing_router is None:
            existing_router = seed_linear_from_gate(original_block.gate)
        self.router = RouterClass(router_config_for(config), existing_router=existing_router, **router_kwargs)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden = hidden_states.shape
        x = hidden_states.view(-1, hidden)                         # [B*T, H] row-major (bsz, position)
        shared = self.shared_expert(x)
        # Our routers are fp32 modules; the 5-tuple's last element is the routing
        # logits actually used (FCVR eval: log of the MC-mean softmax).
        _, _, _, _, logits = self.router(x.float())
        top_v, top_i = logits.topk(self.top_k, dim=-1)
        top_w = torch.softmax(top_v, dim=-1).to(hidden_states.dtype)  # == softmax-all -> top-k -> renormalise
        out = self.experts(x, top_i, top_w)
        out = out + torch.sigmoid(self.shared_expert_gate(x)) * shared
        return out.view(batch_size, seq_len, hidden)


def ensure_containers(model):
    """Idempotently wrap EVERY layer's MoE block in a container whose router is a
    MAP `MoERouter` copy of the native gate. Must run in every entry point (also
    prior_source=pretrained): the native gate's forward returns
    (logits, scores, indices), so the evaluators' `output[-1]` hooks would read
    expert INDICES as logits on any un-swapped layer."""
    causal_model = _decoder(model)
    n = 0
    for layer in causal_model.layers:
        if isinstance(layer.mlp, BayesianQwen35MoeBlock):
            continue
        block = BayesianQwen35MoeBlock(causal_model.config, layer.mlp, MoERouter)
        layer.mlp = block.to(_layer_device(layer))
        n += 1
    if n:
        print(f"--- qwen36: wrapped {n} MoE blocks in BayesianQwen35MoeBlock (MAP router = pre-trained gate) ---")
    return model


# ====================================================
# MAP (Deterministic) Router Functions
# ====================================================
def swap_qwen36_moe_blocks(model):
    """MAP router fine-tuning: containers everywhere, all frozen except the routers."""
    print("--- Preparing model for MAP router tuning ---")
    model = ensure_containers(model)
    for param in model.parameters():
        param.requires_grad = False
    for layer in _decoder(model).layers:
        for param in layer.mlp.router.parameters():
            param.requires_grad = True
    return model


def _map_run_name(args):
    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    map_suffix = getattr(args, "map_suffix", None)
    if map_suffix:
        run_name = f"{run_name}-{map_suffix}"
    return run_name


def save_qwen36_map_routers(model, args):
    print("--- Saving MAP router weights ---")
    output_dir = "./router_weights/base"
    run_name = _map_run_name(args)
    print(f"    -> {os.path.join(output_dir, run_name)}")
    for i, layer in enumerate(_decoder(model).layers):
        layer.mlp.router.save_weights(os.path.join(output_dir, run_name, f"layer_{i}_weights.pt"))


def load_qwen36_map_routers(model, args):
    print("--- Loading base MAP routers ---")
    model = ensure_containers(model)
    causal_model = _decoder(model)
    map_weights_dir = f"./router_weights/base/{_map_run_name(args)}"
    print(f"    <- {map_weights_dir}")
    for i, layer in enumerate(causal_model.layers):
        map_router = MoERouter(config=router_config_for(causal_model.config))
        path = os.path.join(map_weights_dir, f"layer_{i}_weights.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No MAP router weights for layer {i} at {path}. "
                f"Run router-tuning (Stage 2a) for this dataset / --map_suffix first, or use --prior_source pretrained.")
        map_router.load_weights(path, device=_layer_device(layer))
        layer.mlp.router = map_router.to(_layer_device(layer))
    return model


# ====================================================
# Generic Bayesian Router Functions (mirror of granite_adapter)
# ====================================================
def _bayes_run_name(method, args):
    run_name = f"{method}-{args.model_shortcode}-{args.dataset_shortcode}"
    run_suffix = getattr(args, "run_suffix", None)
    if run_suffix:
        run_name = f"{run_name}-{run_suffix}"
    return run_name


def prepare_qwen36_bayesian_routers(model, method, args):
    if method not in ROUTER_CONFIG:
        raise ValueError(f"Unknown Bayesian method: {method}. Supported methods are {list(ROUTER_CONFIG.keys())}")
    config = ROUTER_CONFIG[method]
    RouterClass = config["class"]
    router_kwargs = config["get_kwargs"](args)
    trainable_attrs = config["trainable_attrs"]

    print(f"--- Preparing model for {method.upper()} router tuning ---")
    model = ensure_containers(model)
    causal_model = _decoder(model)
    rcfg = router_config_for(causal_model.config)
    weights_dir = os.path.join(f"./router_weights/{method}", _bayes_run_name(method, args))

    for layer_idx in args.swap_layers:
        layer = causal_model.layers[layer_idx]
        new_router = RouterClass(config=rcfg, existing_router=layer.mlp.router, **router_kwargs)
        if layer_idx in args.load_layers:
            print(f"Loading pre-trained {method.upper()} for layer {layer_idx}...")
            new_router.load_weights(os.path.join(weights_dir, f"layer_{layer_idx}_weights.pt"),
                                    device=_layer_device(layer))
        layer.mlp.router = new_router.to(_layer_device(layer))

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


def save_qwen36_bayesian_routers(model, method, args):
    print(f"--- Saving {method.upper()} router weights ---")
    output_root_dir = f"./router_weights/{method}"
    if method == "vtsr":
        output_root_dir += f"_{args.temperature_mode}"
    save_dir = os.path.join(output_root_dir, _bayes_run_name(method, args))
    causal_model = _decoder(model)
    for layer_idx in args.swap_layers:
        causal_model.layers[layer_idx].mlp.router.save_weights(os.path.join(save_dir, f"layer_{layer_idx}_weights.pt"))


def load_qwen36_bayesian_routers(model, method, args):
    if method not in ROUTER_CONFIG:
        raise ValueError(f"Unknown Bayesian method: {method}. Supported methods are {list(ROUTER_CONFIG.keys())}")
    config = ROUTER_CONFIG[method]
    RouterClass = config["class"]
    router_kwargs = config["get_kwargs"](args)

    print(f"--- Loading pre-trained {method.upper()} routers for evaluation ---")
    model = ensure_containers(model)
    causal_model = _decoder(model)
    rcfg = router_config_for(causal_model.config)
    weights_dir = os.path.join(f"./router_weights/{method}", _bayes_run_name(method, args))
    swap_layers = args.swap_layers if args.swap_layers is not None else range(len(causal_model.layers))

    for layer_idx in swap_layers:
        layer = causal_model.layers[layer_idx]
        new_router = RouterClass(config=rcfg, existing_router=layer.mlp.router, **router_kwargs)
        path = os.path.join(weights_dir, f"layer_{layer_idx}_weights.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained {method.upper()} weights for layer {layer_idx} at {path}. "
                f"Check --run_suffix / --swap_layers / --dataset_shortcode against the training run.")
        new_router.load_weights(path, device=_layer_device(layer))
        layer.mlp.router = new_router.to(_layer_device(layer))
    return model

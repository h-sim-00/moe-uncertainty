"""Gemma 4 26B-A4B (transformers>=5.5 `gemma4`, text stack `gemma4_text`) adapter
-- branch OBQA-gemma.

Mirrors model/adapters/qwen36_adapter.py / granite_adapter.py function-for-
function (same router_weights/ path scheme, same map_suffix / run_suffix
handling, same FileNotFoundError semantics) for a model whose MoE pieces are NOT
grouped in a sub-block: `Gemma4TextDecoderLayer` holds them directly,

    layer.mlp      Gemma4TextMLP      dense MLP, runs on EVERY token (parallel branch)
    layer.router   Gemma4TextRouter   norm(no scale) -> * scale * H^-0.5 -> proj Linear(H, E, no bias)
                                      -> softmax -> top-k -> renormalise -> * per_expert_scale[top-k]
                                      returns (router_probabilities, top_k_weights, top_k_index)
    layer.experts  Gemma4TextExperts  fused gate_up_proj [E, 2I, H], down_proj [E, H, I]

and combines them inside its forward as
    norm_1(mlp(pre_norm(x))) + norm_2(experts(pre_norm_2(x), top_k_index, top_k_weights)).
The layer calls `self.router(hidden_flat)` with hidden_flat = residual.reshape(-1, H)
and only consumes (top_k_weights, top_k_index), so the Bayesian swap replaces
`layer.router` with `BayesianGemma4Router`: a container that keeps Gemma's frozen
pre-processing (norm, scale, per_expert_scale) and delegates the routing LOGITS
to one of OUR router classes under the attribute name `.router` (which is what
`model.adapters.moe_router(layer)` returns, so every evaluator / hook works).

Same two deliberate choices as the Qwen port:
  * routing is derived from the `logits` the inner router RETURNS (FCVR eval:
    the MC-averaged log-mean-probs), never from a side attribute;
  * the inner routers are fp32 modules seeded from the bf16 `proj`, fed the
    normalised+scaled router input `u.float()` -- Cholesky / MVN sampling / KL
    stay fp32 exactly as on Granite. `u` (not the raw residual) is the "hidden
    state" the FCVR backbone and KL see, because that is the pre-trained
    router's input space.
Top-k-then-softmax over the selected logits equals Gemma's softmax-all -> top-k
-> renormalise; per_expert_scale is re-applied afterwards exactly as natively, so
the MAP container reproduces the native router up to bf16-vs-fp32 rounding of
the projection (checked by smoke-gemma4-model.py).
"""
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from model.routers.base import MoERouter
from model.adapters.granite_adapter import ROUTER_CONFIG   # same method registry


# ====================================================
# Config / gate shims
# ====================================================
def router_config_for(config):
    """The router classes read Granite's names (num_local_experts,
    num_experts_per_tok); Gemma4TextConfig calls them num_experts / top_k_experts."""
    if hasattr(config, "num_local_experts"):
        return config
    return SimpleNamespace(
        hidden_size=config.hidden_size,
        num_local_experts=config.num_experts,
        num_experts_per_tok=config.top_k_experts,
    )


def seed_linear_from_proj(proj):
    """MoERouter / FCVR copy `existing_router.layer.state_dict()` into an
    nn.Linear(H, E, bias=False). Gemma's `proj` IS such a Linear (bf16); hand
    over an fp32 copy under the expected attribute name."""
    lin = nn.Linear(proj.in_features, proj.out_features, bias=False)
    with torch.no_grad():
        lin.weight.copy_(proj.weight.detach().float())
    return SimpleNamespace(layer=lin)


def _decoder(model):
    return model.base_model.model.model


def _layer_device(layer):
    return next(layer.parameters()).device


def _container(layer):
    c = layer.router
    if not isinstance(c, BayesianGemma4Router):
        raise AttributeError("gemma4: layer.router is not a BayesianGemma4Router -- run ensure_containers first")
    return c


# ====================================================
# Container router
# ====================================================
class BayesianGemma4Router(nn.Module):
    """Drop-in replacement for Gemma4TextRouter with a pluggable (Bayesian) router
    for the logits. Keeps the native, frozen pre-processing parameters."""

    def __init__(self, config, original_router, RouterClass=MoERouter, existing_router=None, **router_kwargs):
        super().__init__()
        self.norm = original_router.norm                          # Gemma4RMSNorm(with_scale=False): no parameters
        self.scale = original_router.scale                        # nn.Parameter [H]
        self.per_expert_scale = original_router.per_expert_scale  # nn.Parameter [E]
        self.scalar_root_size = original_router.scalar_root_size  # H ** -0.5
        self.top_k = config.top_k_experts
        self.num_experts = config.num_experts
        if existing_router is None:
            existing_router = seed_linear_from_proj(original_router.proj)
        self.router = RouterClass(router_config_for(config), existing_router=existing_router, **router_kwargs)

    def router_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gemma4TextRouter's pre-processing, same dtype path as the native module,
        then fp32 for our routers. hidden_states: [N, H] (row-major over (bsz, pos))."""
        u = self.norm(hidden_states)
        u = u * self.scale * self.scalar_root_size
        return u.float()

    def forward(self, hidden_states: torch.Tensor):
        u = self.router_input(hidden_states)
        _, _, _, _, logits = self.router(u)                        # [N, E] fp32 (FCVR eval: log MC-mean probs)
        router_probabilities = torch.softmax(logits, dim=-1)       # returned like the native router (unused by the layer)
        top_v, top_i = logits.topk(self.top_k, dim=-1)
        top_w = torch.softmax(top_v, dim=-1)                       # == softmax-all -> top-k -> renormalise
        top_w = top_w * self.per_expert_scale[top_i].float()       # native: fp32 weights * per_expert_scale -> fp32
        return router_probabilities, top_w, top_i


def ensure_containers(model):
    """Idempotently wrap EVERY MoE layer's router in a container whose inner router
    is a MAP `MoERouter` copy of the native `proj`. Must run in every entry point
    (also prior_source=pretrained): the native router's forward returns
    (probabilities, weights, indices), so the evaluators' `output[-1]` hooks would
    read expert INDICES as logits on any un-swapped layer."""
    causal_model = _decoder(model)
    n = 0
    for layer in causal_model.layers:
        if not getattr(layer, "enable_moe_block", False):
            continue
        if isinstance(layer.router, BayesianGemma4Router):
            continue
        container = BayesianGemma4Router(causal_model.config, layer.router, MoERouter)
        layer.router = container.to(_layer_device(layer))
        n += 1
    if n:
        print(f"--- gemma4: wrapped {n} MoE routers in BayesianGemma4Router (MAP router = pre-trained proj) ---")
    return model


# ====================================================
# MAP (Deterministic) Router Functions
# ====================================================
def swap_gemma4_moe_blocks(model):
    """MAP router fine-tuning: containers everywhere, all frozen except the inner
    routers (the projection; Gemma's norm scale / per_expert_scale stay frozen)."""
    print("--- Preparing model for MAP router tuning ---")
    model = ensure_containers(model)
    for param in model.parameters():
        param.requires_grad = False
    for layer in _decoder(model).layers:
        for param in _container(layer).router.parameters():
            param.requires_grad = True
    return model


def _map_run_name(args):
    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    map_suffix = getattr(args, "map_suffix", None)
    if map_suffix:
        run_name = f"{run_name}-{map_suffix}"
    return run_name


def save_gemma4_map_routers(model, args):
    print("--- Saving MAP router weights ---")
    output_dir = "./router_weights/base"
    run_name = _map_run_name(args)
    print(f"    -> {os.path.join(output_dir, run_name)}")
    for i, layer in enumerate(_decoder(model).layers):
        _container(layer).router.save_weights(os.path.join(output_dir, run_name, f"layer_{i}_weights.pt"))


def load_gemma4_map_routers(model, args):
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
        _container(layer).router = map_router.to(_layer_device(layer))
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


def prepare_gemma4_bayesian_routers(model, method, args):
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
        new_router = RouterClass(config=rcfg, existing_router=_container(layer).router, **router_kwargs)
        if layer_idx in args.load_layers:
            print(f"Loading pre-trained {method.upper()} for layer {layer_idx}...")
            new_router.load_weights(os.path.join(weights_dir, f"layer_{layer_idx}_weights.pt"),
                                    device=_layer_device(layer))
        _container(layer).router = new_router.to(_layer_device(layer))

    for param in model.parameters():
        param.requires_grad = False
    print(f"Unfreezing routers in layers: {args.train_layers}")
    for layer_idx in args.train_layers:
        router = _container(causal_model.layers[layer_idx]).router
        if trainable_attrs is None:
            for param in router.parameters():
                param.requires_grad = True
        else:
            for attr_name in trainable_attrs:
                for param in getattr(router, attr_name).parameters():
                    param.requires_grad = True
    return model


def save_gemma4_bayesian_routers(model, method, args):
    print(f"--- Saving {method.upper()} router weights ---")
    output_root_dir = f"./router_weights/{method}"
    if method == "vtsr":
        output_root_dir += f"_{args.temperature_mode}"
    save_dir = os.path.join(output_root_dir, _bayes_run_name(method, args))
    causal_model = _decoder(model)
    for layer_idx in args.swap_layers:
        _container(causal_model.layers[layer_idx]).router.save_weights(
            os.path.join(save_dir, f"layer_{layer_idx}_weights.pt"))


def load_gemma4_bayesian_routers(model, method, args):
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
        new_router = RouterClass(config=rcfg, existing_router=_container(layer).router, **router_kwargs)
        path = os.path.join(weights_dir, f"layer_{layer_idx}_weights.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained {method.upper()} weights for layer {layer_idx} at {path}. "
                f"Check --run_suffix / --swap_layers / --dataset_shortcode against the training run.")
        new_router.load_weights(path, device=_layer_device(layer))
        _container(layer).router = new_router.to(_layer_device(layer))
    return model

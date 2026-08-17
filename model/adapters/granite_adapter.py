# Granite MoE's adapter:
# Base Router is implemented on top of Granite MoE, so no need for adapter.
# But include two APIs for replacing routers;
from model.routers.mcdr import MCDropoutRouter
from model.routers.mfvr import MeanFieldVariationalRouter
from model.routers.fcvr import FullCovarianceVariationalRouter
from model.routers.vtsr import VariationalTemperatureRouter
from model.routers.base import MoERouter
import os

ROUTER_CONFIG = {
    "mcdr": {
        "class": MCDropoutRouter,
        "get_kwargs": lambda args: {"dropout_rate": args.dropout_rate},
        "trainable_attrs": None  # None signifies all parameters are trainable
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
# MAP (Deterministic) Router Functions
# ====================================================

# (1) Initilisation: For MAP det router finetuning
def swap_granite_moe_blocks(model):
    """
    Initializes the model for MAP router fine-tuning.
    Replaces all original routers with the base MoERouter and unfreezes them.
    """
    print("--- Preparing model for MAP router tuning ---")
    for param in model.parameters():
        param.requires_grad = False

    causal_model = model.base_model.model.model
    for layer in causal_model.layers:
        new_router = MoERouter(config=causal_model.config, existing_router=layer.block_sparse_moe.router)
        layer.block_sparse_moe.router = new_router.to(model.device)
        for param in new_router.parameters():
            param.requires_grad = True
    
    return model

# (2) Saving: save finetuned map weights
def save_granite_map_routers(model, args):
    """Saves the fine-tuned MAP router weights."""
    print("--- Saving MAP router weights ---")
    output_dir = "./router_weights/base"
    causal_model = model.base_model.model.model
    run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    # Optional suffix so a MAP run never overwrites an earlier one (mirrors
    # run_suffix for the Bayesian weights). Default path unchanged.
    map_suffix = getattr(args, "map_suffix", None)
    if map_suffix:
        run_name = f"{run_name}-{map_suffix}"
    print(f"    -> {os.path.join(output_dir, run_name)}")
    for i, layer in enumerate(causal_model.layers):
        save_path = os.path.join(output_dir, run_name, f"layer_{i}_weights.pt")
        layer.block_sparse_moe.router.save_weights(save_path)

# (3) Loading: load finetuned map weights
def load_granite_map_routers(model, args):
    """Loads pre-trained MAP router weights into the model."""
    print("--- Loading base MAP routers ---")
    causal_model = model.base_model.model.model
    map_run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    map_suffix = getattr(args, "map_suffix", None)
    if map_suffix:
        map_run_name = f"{map_run_name}-{map_suffix}"
    map_weights_dir = f"./router_weights/base/{map_run_name}"
    print(f"    <- {map_weights_dir}")

    for i, layer in enumerate(causal_model.layers):
        map_router = MoERouter(config=causal_model.config)
        map_weights_path = os.path.join(map_weights_dir, f"layer_{i}_weights.pt")
        if not os.path.exists(map_weights_path):
            raise FileNotFoundError(
                f"No MAP router weights for layer {i} at {map_weights_path}. "
                f"Run router-tuning (Stage 2a) for this dataset / --map_suffix first, or use --prior_source pretrained.")
        map_router.load_weights(map_weights_path, device=model.device)
        layer.block_sparse_moe.router = map_router.to(model.device)

    return model

# ====================================================
# Generic Bayesian Router Functions (New Elegant API)
# ====================================================

# (4) Preparing for Bayeisan Tuning
def prepare_granite_bayesian_routers(model, method, args):
    """
    A generic function to prepare the model for Bayesian router training.
    Handles swapping, loading, freezing, and unfreezing for any specified method.
    """
    if method not in ROUTER_CONFIG:
        raise ValueError(f"Unknown Bayesian method: {method}. Supported methods are {list(ROUTER_CONFIG.keys())}")

    config = ROUTER_CONFIG[method]
    RouterClass = config["class"]
    router_kwargs = config["get_kwargs"](args)
    trainable_attrs = config["trainable_attrs"]

    print(f"--- Preparing model for {method.upper()} router tuning ---")

    output_root_dir = f"./router_weights/{method}"
    run_name = f"{method}-{args.model_shortcode}-{args.dataset_shortcode}"
    # Mirror the optional suffix used at save time, so --load_layers resumes
    # from THIS run's weights dir instead of the unsuffixed one.
    run_suffix = getattr(args, "run_suffix", None)
    if run_suffix:
        run_name = f"{run_name}-{run_suffix}"

    causal_model = model.base_model.model.model

    # 1. Swap & Load
    for layer_idx in args.swap_layers:
        target_layer = causal_model.layers[layer_idx]
        new_router = RouterClass(
            config=causal_model.config,
            existing_router=target_layer.block_sparse_moe.router,
            **router_kwargs
        )
        if layer_idx in args.load_layers:
            print(f"Loading pre-trained {method.upper()} for layer {layer_idx}...")
            weights_path = os.path.join(output_root_dir, run_name, f"layer_{layer_idx}_weights.pt")
            new_router.load_weights(weights_path, device=model.device)

        target_layer.block_sparse_moe.router = new_router.to(model.device)

    # 2. Freeze & Unfreeze
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Unfreezing routers in layers: {args.train_layers}")
    for layer_idx in args.train_layers:
        router = causal_model.layers[layer_idx].block_sparse_moe.router
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
def save_granite_bayesian_routers(model, method, args):
    """A generic function to save the weights of trained Bayesian routers."""
    print(f"--- Saving {method.upper()} router weights ---")
    output_root_dir = f"./router_weights/{method}"
    if method == 'vtsr':
        output_root_dir += f"_{args.temperature_mode}"
    run_name = f"{method}-{args.model_shortcode}-{args.dataset_shortcode}"
    # Optional suffix to keep independent runs (e.g. non-progressive on a
    # different layer set) from overwriting each other's weights.
    run_suffix = getattr(args, "run_suffix", None)
    if run_suffix:
        run_name = f"{run_name}-{run_suffix}"
    save_dir = os.path.join(output_root_dir, run_name)

    causal_model = model.base_model.model.model

    # Save all swapped layers to preserve state for progressive training
    for layer_idx in args.swap_layers:
        save_path = os.path.join(save_dir, f"layer_{layer_idx}_weights.pt")
        causal_model.layers[layer_idx].block_sparse_moe.router.save_weights(save_path)

# (6) Load Bayesian Parameters
def load_granite_bayesian_routers(model, method, args):
    """A generic function to load pre-trained Bayesian routers for evaluation."""
    if method not in ROUTER_CONFIG:
        raise ValueError(f"Unknown Bayesian method: {method}. Supported methods are {list(ROUTER_CONFIG.keys())}")

    config = ROUTER_CONFIG[method]
    RouterClass = config["class"]
    router_kwargs = config["get_kwargs"](args)

    print(f"--- Loading pre-trained {method.upper()} routers for evaluation ---")
    
    output_root_dir = f"./router_weights/{method}"
    run_name = f"{method}-{args.model_shortcode}-{args.dataset_shortcode}"
    # Mirror the optional suffix used at save time.
    run_suffix = getattr(args, "run_suffix", None)
    if run_suffix:
        run_name = f"{run_name}-{run_suffix}"
    weights_dir = os.path.join(output_root_dir, run_name)

    causal_model = model.base_model.model.model

    # Swap layers specified in args, or all layers if not specified
    swap_layers = args.swap_layers if args.swap_layers is not None else range(len(causal_model.layers))

    for layer_idx in swap_layers:
        target_layer = causal_model.layers[layer_idx]
        new_router = RouterClass(
            config=causal_model.config,
            existing_router=target_layer.block_sparse_moe.router,
            **router_kwargs
        )
        weights_path = os.path.join(weights_dir, f"layer_{layer_idx}_weights.pt")
        if not os.path.exists(weights_path):
            # Hard error (was a print warning): a freshly-initialised router
            # emits a near-constant uncertainty signal, so a silent fallback
            # here produces a full, plausible-looking eval run with no trained
            # signal in it. Fail loudly instead.
            raise FileNotFoundError(
                f"No trained {method.upper()} weights for layer {layer_idx} at {weights_path}. "
                f"Check --run_suffix / --swap_layers / --dataset_shortcode against the training run."
            )
        new_router.load_weights(weights_path, device=model.device)

        target_layer.block_sparse_moe.router = new_router.to(model.device)
    
    return model
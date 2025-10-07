# Granite MoE's adapter:
# Base Router is implemented on top of Granite MoE, so no need for adapter.
# But include two APIs for replacing routers;

from model.routers.base import MoERouter
import os

# Initilisation: For MAP det router finetuning
def swap_granite_moe_blocks(model):
    for param in model.parameters():
        param.requires_grad = False
    causal_model = model.base_model.model.model
    for layer in causal_model.layers:
        new_router = MoERouter(config=causal_model.config, existing_router=layer.block_sparse_moe.router)
        new_router.to(model.device)
        layer.block_sparse_moe.router = new_router
        for param in new_router.parameters():
            param.requires_grad = True
    return model

# Loading: load finetuned map weights
def load_granite_map_routers(model, args):
    print("--- Loading base MAP routers ---")
    causal_model = model.base_model.model.model
    map_run_name = f"{args.model_shortcode}_{args.dataset_shortcode}"
    map_weights_dir = f"./router_weights/base/{map_run_name}"
    
    for i, layer in enumerate(causal_model.layers):
        map_router = MoERouter(config=causal_model.config)
        map_weights_path = os.path.join(map_weights_dir, f"layer_{i}_weights.pt")
        map_router.load_weights(map_weights_path, device=model.device)
        map_router.to(model.device)
        layer.block_sparse_moe.router = map_router

    return model


# Bayesianfying: For Bayesian training
# swap, load, freeze & unfreeze
def prepare_granite_bayesian_routers(model, args):
    output_root_dir = "./router_weights/mcdr"
    run_name = f"mcdr-{args.model_shortcode}-{args.dataset_shortcode}"
    output_dir = os.path.join(output_root_dir, run_name)
    causal_model = model.base_model.model.model

    # Swap & Load
    for layer_idx in args.swap_layers:
        target_layer = causal_model.layers[layer_idx]
        
        if layer_idx in args.load_layers:
            # This layer should have an MCDR with pre-trained weights
            print(f"Loading pre-trained MCDR for layer {layer_idx}...")
            new_router = MCDropoutRouter(config=causal_model.config, dropout_rate=args.dropout_rate)
            weights_path = os.path.join(output_dir, f"layer_{layer_idx}_weights.pt")
            new_router.load_weights(weights_path, device=model.device)
        else:
            # This layer should have a new MCDR initialized from the MAP router
            print(f"Initializing new MCDR for layer {layer_idx} from MAP weights...")
            new_router = MCDropoutRouter(
                config=causal_model.config,
                existing_router=target_layer.block_sparse_moe.router,
                dropout_rate=args.dropout_rate
            )
        
        new_router.to(model.device)
        target_layer.block_sparse_moe.router = new_router

    # Freeze & Unfreeze
    for param in model.parameters():
        param.requires_grad = False
    
    for layer_idx in args.train_layers:
        print(f"Unfreezing router in layer {layer_idx} for training.")
        for param in causal_model.layers[layer_idx].block_sparse_moe.router.parameters():
            param.requires_grad = True

    return model

def save_granite_routers(model, args):
    
    output_root_dir = "./router_weights/mcdr"
    run_name = f"mcdr-{args.model_shortcode}-{args.dataset_shortcode}"
    output_dir = os.path.join(output_root_dir, run_name)
    causal_model = model.base_model.model.model

    for layer_idx in args.train_layers:
        save_path = os.path.join(output_dir, f"layer_{layer_idx}_weights.pt")
        causal_model.layers[layer_idx].block_sparse_moe.router.save_weights(save_path)
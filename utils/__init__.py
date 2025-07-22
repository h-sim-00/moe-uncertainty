import os
import wandb
from huggingface_hub import login as hf_login

def setup_environment():
    """Configures HuggingFace cache and logs into services."""
    # These paths are specific to your cluster environment
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if not IN_COLAB:
        print("Setting up new cache paths for huggingface and Wandb...")
        
        HF_HOME = "/vol/bitbucket/al1624/.cache/huggingface"
        HF_DATASETS_CACHE = "/vol/bitbucket/al1624/.cache/huggingface/datasets"
        WANDB_STORAGE_DIR = "/vol/bitbucket/al1624/.cache/wandb"

        os.makedirs(WANDB_STORAGE_DIR, exist_ok=True)

        os.environ['HF_HOME'] = HF_HOME
        os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
        os.environ['WANDB_DIR'] = WANDB_STORAGE_DIR

    # Login to WandB and HuggingFace
    print("Logging into WandB and HuggingFace...")
    WANDB_KEY = "8d44174f1416d56dc5470b57deb50339b19f22e7"
    HF_TOKEN = "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"
    wandb.login(key=WANDB_KEY)
    hf_login(token=HF_TOKEN)
    print("Login successful.")
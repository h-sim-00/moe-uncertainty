# filename: debug_python_path.py
import os
from pathlib import Path
import sys

print("--- STARTING PYTHON DIAGNOSTIC SCRIPT ---")

# --- Step 1: Check the environment BEFORE importing huggingface_hub ---
print("\n[Step 1] Checking environment at the very start...")
initial_hf_home = os.environ.get('HF_HOME')
print(f"  Initial os.environ.get('HF_HOME') ---> '{initial_hf_home}'")

# --- Step 2: Set the environment variable ---
print("\n[Step 2] Setting HF_HOME environment variable within Python...")
TARGET_CACHE_DIR = "/vol/bitbucket/al1624/.cache/huggingface"
os.environ['HF_HOME'] = TARGET_CACHE_DIR
print(f"  os.environ['HF_HOME'] has been set to ---> '{os.environ.get('HF_HOME')}'")

# --- Step 3: NOW import huggingface_hub and check its resolved path ---
# This is the most critical test. It tells us if the library sees the variable we just set.
print("\n[Step 3] Importing huggingface_hub and checking its constants...")
try:
    from huggingface_hub import constants
    # The HUGGINGFACE_HUB_CACHE constant is what the library uses internally.
    resolved_cache_path = constants.HUGGINGFACE_HUB_CACHE
    print(f"  huggingface_hub.constants.HUGGINGFACE_HUB_CACHE is ---> '{resolved_cache_path}'")
    
    if TARGET_CACHE_DIR in str(resolved_cache_path):
        print("\n  [✓] SUCCESS: The library correctly resolved the custom cache path.")
    else:
        print("\n  [!] FAILURE: The library is still pointing to the default path.")

except Exception as e:
    print(f"\n  [!] ERROR: Failed to import or check huggingface_hub constants: {e}")

# --- Step 4: Final check with a write test ---
print("\n[Step 4] Performing a write test to the target directory...")
try:
    target_dir_path = Path(TARGET_CACHE_DIR)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    test_file = target_dir_path / "permission_test.tmp"
    with open(test_file, "w") as f:
        f.write("success")
    os.remove(test_file)
    print(f"  [✓] SUCCESS: Write permissions are correct for '{TARGET_CACHE_DIR}'")
except Exception as e:
    print(f"  [!] FAILURE: Could not write to the target directory. Error: {e}")

print("\n--- DIAGNOSTIC SCRIPT FINISHED ---")

from transformers import Qwen2MoeForCausalLM
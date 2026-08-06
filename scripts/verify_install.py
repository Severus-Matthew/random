"""Quick sanity check that the environment is set up correctly."""
import sys

def check(name, fn):
    try:
        fn()
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

print("Checking environment...")
check("torch",         lambda: __import__("torch"))
check("transformers",  lambda: __import__("transformers"))
check("vllm",          lambda: __import__("vllm"))
check("trl",           lambda: __import__("trl"))
check("ray",           lambda: __import__("ray"))
check("datasets",      lambda: __import__("datasets"))
check("wandb",         lambda: __import__("wandb"))
check("accelerate",    lambda: __import__("accelerate"))

import torch
print(f"\nPyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version    : {torch.version.cuda}")
print("\nDone.")

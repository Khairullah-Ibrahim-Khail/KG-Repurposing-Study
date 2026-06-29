"""One-shot dependency installer (GPU build).

    python scripts/install_deps.py

Order matters: CUDA PyTorch first, then PyG, then the rest.
"""

import importlib
import subprocess
import sys

TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
PYG_INDEX = "https://data.pyg.org/whl/torch-2.5.0+cu124.html"


def pip(*args) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(">>>", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"install failed: {args}")


def main() -> None:
    pip("torch==2.5.0", "torchvision==0.20.0", "--index-url", TORCH_INDEX)
    pip("torch-geometric", "--index-url", PYG_INDEX)
    pip("scikit-learn", "pandas", "numpy", "pyyaml", "matplotlib", "tqdm")

    print("\n=== verify ===")
    for pkg in ("torch", "torch_geometric", "sklearn", "pandas", "numpy"):
        try:
            mod = importlib.import_module(pkg)
            print(f"  {pkg}: {getattr(mod, '__version__', '?')}")
        except ImportError as exc:
            print(f"  {pkg}: MISSING ({exc})")
    import torch
    print(f"\nCUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()

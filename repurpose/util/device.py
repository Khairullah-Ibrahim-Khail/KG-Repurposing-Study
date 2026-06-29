"""CUDA device setup. This project trains on GPU only and refuses to fall back
to CPU silently — call ``require_cuda()`` at the top of every runner.
"""

import os

import torch


def require_cuda(device_id: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the NVIDIA driver and the PyTorch build; "
            "this project requires a GPU."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device(f"cuda:{device_id}")
    props = torch.cuda.get_device_properties(device_id)
    print(f"[device] {props.name} | {props.total_memory / 1e9:.1f} GB | cuda:{device_id}")
    return device

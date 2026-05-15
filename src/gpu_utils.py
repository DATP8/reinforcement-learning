"""
GPU Training Utilities
======================

Helper functions for GPU-accelerated RL training with mixed precision support.
Includes device management, CUDA optimizations, and memory utilities.
"""

import torch
import torch.nn as nn
from typing import Optional


class MixedPrecisionWrapper:
    """
    Wrapper for mixed precision training (float32 forward, float16 backward).
    Reduces memory usage and speeds up training on GPUs that support it.
    """

    def __init__(self, model: nn.Module, enabled: bool = True):
        self.model = model
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.enabled else None

    def forward(self, *args, **kwargs):
        """Forward pass with optional automatic mixed precision."""
        if self.enabled:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                return self.model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)

    def backward(self, loss: torch.Tensor):
        """Backward pass with optional gradient scaling for mixed precision."""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        """Optimizer step with optional gradient unscaling."""
        if self.enabled:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimizer.step()


def get_device() -> torch.device:
    """
    Get the preferred device (GPU if available, else CPU).

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_cuda_optimizations():
    """
    Enable CUDA-specific optimizations for faster training.

    Enables:
    - cuDNN auto-tuner for convolution operations (benchmark=True)
    - TF32 precision for matmul on compatible GPUs (A100, etc.)
      This has minimal accuracy impact for training but provides significant speedup.
    """
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner for convolution operations
        torch.backends.cudnn.benchmark = True

        # Use TF32 precision on compatible GPUs (A100, etc.) for faster matmul
        # This has minimal accuracy impact for training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def should_pin_memory() -> bool:
    """
    Check if pinned memory should be used for DataLoaders.

    Pinned memory allows faster CPU-GPU transfers on CUDA devices.
    Only beneficial when GPU is actively used.

    Returns
    -------
    bool
        True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def log_gpu_memory(prefix: str = ""):
    """
    Log current GPU memory usage if CUDA is available.

    Parameters
    ----------
    prefix : str
        Optional prefix to add to the log message.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        msg = f"{prefix} GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
        print(msg)


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_info() -> dict:
    """
    Get detailed GPU information.

    Returns
    -------
    dict
        Dictionary with GPU info including device count, name, and memory.
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }

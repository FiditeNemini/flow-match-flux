"""
Main training script for the Chroma model.

This script initializes and runs the training process for the Chroma model,
supporting multiple hardware platforms (CUDA, MPS, CPU).

It reads the `device` setting from `config.json`.
- If 'cuda' is selected, it uses `torch.multiprocessing.spawn` to launch the
  `train_chroma` function on multiple GPU processes.
- If 'mps' or 'cpu' is selected, it runs on a single process.

The configuration for the training is expected to be in a `config.json` file
in the same directory as this script.
"""
import torch.multiprocessing as mp
import os
import torch
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer.train_chroma import train_chroma
if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)

    device_setting = config["training"].get("device", "auto").lower()

    device = "cpu"
    if device_setting == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    elif device_setting in ["cuda", "mps", "cpu"]:
        # Ensure the selected device is available
        if device_setting == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is selected but not available.")
        if device_setting == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is selected but not available.")
        device = device_setting
    else:
        raise ValueError(f"Invalid device setting: {device_setting}")

    if device == "cuda":
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    # Use spawn method for starting processes
    mp.spawn(
        train_chroma,
        args=(world_size, device),
        nprocs=world_size,
        join=True
    )
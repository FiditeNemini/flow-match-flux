"""
Training script for the Chroma model with Classifier-Free Guidance (CFG).

This script launches the multi-GPU training process for the Chroma model, specifically
using a CFG-enabled training function `train_chroma_cfg`. It leverages
`torch.multiprocessing.spawn` to distribute the training across all available GPUs.

The configuration for this training run should be specified in `config_cfg.json`.
"""
import torch.multiprocessing as mp
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer.train_chroma_cfg import train_chroma
if __name__ == "__main__":

    world_size = torch.cuda.device_count()

    # Use spawn method for starting processes
    mp.spawn(
        train_chroma,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
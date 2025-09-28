"""
Main training script for the Lumina model.

This script initializes and runs the multi-GPU training process for the Lumina model.
It uses the `torch.multiprocessing.spawn` function to launch the `train_lumina`
function on multiple GPU processes. The number of processes is determined by the
number of available CUDA devices.

The configuration for the training is expected to be in a `config.json` file
in the same directory as this script.
"""
import torch.multiprocessing as mp
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer.train_lumina import train_lumina
if __name__ == "__main__":

    world_size = torch.cuda.device_count()

    # Use spawn method for starting processes
    mp.spawn(
        train_lumina,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
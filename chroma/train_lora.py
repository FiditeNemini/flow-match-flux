"""
Training script for fine-tuning the Chroma model using Low-Rank Adaptation (LoRA).

This script initiates a multi-GPU training process tailored for LoRA fine-tuning.
It uses the `train_chroma_lora` function and distributes the workload across
all available GPUs using `torch.multiprocessing.spawn`.

The specific LoRA configuration, along with other training parameters, should be
defined in the `config_lora.json` file.
"""
import torch.multiprocessing as mp
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer.train_chroma_lora import train_chroma
if __name__ == "__main__":

    world_size = torch.cuda.device_count()

    # Use spawn method for starting processes
    mp.spawn(
        train_chroma,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
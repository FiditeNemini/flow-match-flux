import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from tqdm import tqdm
from safetensors.torch import safe_open, save_file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from torchastic import Compass, StochasticAccumulator
import random

from transformers import T5Tokenizer
import wandb

from src.dataloaders.dataloader import TextImageDataset
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.sampling import get_noise, get_schedule, denoise_cfg
from src.models.chroma.utils import (
    vae_flatten,
    prepare_latent_image_ids,
    vae_unflatten,
    calculate_shift,
    time_shift,
)
from src.models.chroma.module.autoencoder import AutoEncoder, ae_params
from src.math_utils import cosine_optimal_transport
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors
import src.lora_and_quant as lora_and_quant

from huggingface_hub import HfApi, upload_file
import time

from aim import Run, Image as AimImage
from datetime import datetime
from PIL import Image as PILImage

@dataclass
class TrainingConfig:
    master_seed: int
    cache_minibatch: int
    train_minibatch: int
    offload_param_count: int
    lr: float
    weight_decay: float
    warmup_steps: int
    change_layer_every: int
    trained_single_blocks: int
    trained_double_blocks: int
    save_every: int
    save_folder: str
    contrastive_strength: str
    aim_path: Optional[str] = None
    aim_experiment_name: Optional[str] = None
    aim_hash: Optional[str] = None
    aim_steps: Optional[int] = 0
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    


@dataclass
class InferenceConfig:
    inference_every: int
    inference_folder: str
    steps: int
    guidance: int
    cfg: int
    prompts: list[str]
    first_n_steps_wo_cfg: int
    image_dim: tuple[int, int]
    t5_max_length: int


@dataclass
class DataloaderConfig:
    batch_size: int
    jsonl_metadata_path: str
    image_folder_path: str
    base_resolution: list[int]
    shuffle_tags: bool
    tag_drop_percentage: float
    uncond_percentage: float
    resolution_step: int
    num_workers: int
    prefetch_factor: int
    ratio_cutoff: float
    thread_per_worker: int


@dataclass
class ModelConfig:
    """Dataclass to store model paths."""

    chroma_path: str
    vae_path: str
    t5_path: str
    t5_config_path: str
    t5_tokenizer_path: str
    t5_to_8bit: bool
    t5_max_length: int


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def create_distribution(num_points, device=None):
    # Probability range on x axis
    x = torch.linspace(0, 1, num_points, device=device)

    # Custom probability density function
    probabilities = -7.7 * ((x - 0.5) ** 2) + 2

    # Normalize to sum to 1
    probabilities /= probabilities.sum()

    return x, probabilities


# Upload the model to Hugging Face Hub
def upload_to_hf(model_filename, path_in_repo, repo_id, token, max_retries=3):
    api = HfApi()

    for attempt in range(max_retries):
        try:
            upload_file(
                path_or_fileobj=model_filename,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
            )
            print(f"Model uploaded to {repo_id}/{path_in_repo}")
            return  # Exit function if successful

        except Exception as e:
            print(f"Upload attempt {attempt + 1} failed: {e}")
            time.sleep(2**attempt)  # Exponential backoff

    print("Upload failed after multiple attempts.")


def sample_from_distribution(x, probabilities, num_samples, device=None):
    # Step 1: Compute the cumulative distribution function
    cdf = torch.cumsum(probabilities, dim=0)

    # Step 2: Generate uniform random samples
    uniform_samples = torch.rand(num_samples, device=device)

    # Step 3: Map uniform samples to the x values using the CDF
    indices = torch.searchsorted(cdf, uniform_samples, right=True)

    # Get the corresponding x values for the sampled indices
    sampled_values = x[indices]

    return sampled_values

def one_hot_presence_tensor(str_list):
    """
    Converts a list of strings to a tensor where:
    - 0 indicates an empty string
    - 1 indicates a non-empty string
    """
    return torch.tensor([0.0 if s == "" else 1.0 for s in str_list], dtype=torch.bfloat16)

def prepare_sot_pairings(latents):
    # stochastic optimal transport pairings
    # just use mean because STD is so small and practically negligible
    latents = latents.to(torch.float32)
    latents, latent_shape = vae_flatten(latents)
    n, c, h, w = latent_shape
    image_pos_id = prepare_latent_image_ids(n, h, w)

    # randomize ode timesteps
    # input_timestep = torch.round(
    #     F.sigmoid(torch.randn((n,), device=latents.device)), decimals=3
    # )
    num_points = 1000  # Number of points in the range
    x, probabilities = create_distribution(num_points, device=latents.device)
    input_timestep = sample_from_distribution(
        x, probabilities, n, device=latents.device
    )

    # biasing towards earlier more noisy steps where it's the most uncertain
    # input_timestep = time_shift(0.5, 1, input_timestep)

    timesteps = input_timestep[:, None, None]
    # 1 is full noise 0 is full image
    noise = torch.randn_like(latents)

    # compute OT pairings
    transport_cost, indices = cosine_optimal_transport(
        latents.reshape(n, -1), noise.reshape(n, -1)
    )
    noise = noise[indices[1].view(-1)]

    # random lerp points
    noisy_latents = latents * (1 - timesteps) + noise * timesteps

    # target vector that being regressed on
    target = noise - latents

    return noisy_latents, target, input_timestep, image_pos_id, latent_shape


def init_optimizer(model, trained_layer_keywords, lr, wd, warmup_steps):
    # TODO: pack this into a function
    trained_params = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trained_layer_keywords):
            param.requires_grad = True
            trained_params.append((name, param))
        else:
            param.requires_grad = False  # Optionally disable grad for others
    # return hooks so it can be released later on
    hooks = StochasticAccumulator.assign_hooks(model)


    return None, None, hooks, trained_params


def synchronize_gradients(model, scale=1):
    for param in model.parameters():
        if param.grad is not None:
            # Synchronize gradients by summing across all processes
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Average the gradients if needed
            if scale > 1:
                param.grad /= scale


def save_part(model, trained_layer_keywords, counter, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    full_state_dict = model.state_dict()

    filtered_state_dict = {}
    for k, v in full_state_dict.items():
        if any(keyword in k for keyword in trained_layer_keywords):
            filtered_state_dict[k] = v

    torch.save(
        filtered_state_dict, os.path.join(save_folder, f"trained_part_{counter}.pth")
    )


def cast_linear(module, dtype):
    """
    Recursively cast all nn.Linear layers in the model to bfloat16.
    """
    for name, child in module.named_children():
        # If the child module is nn.Linear, cast it to bf16
        if isinstance(child, nn.Linear):
            child.to(dtype)
        else:
            # Recursively apply to child modules
            cast_linear(child, dtype)


def save_config_to_json(filepath: str, **configs):
    json_data = {key: asdict(value) for key, value in configs.items()}
    with open(filepath, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def dump_dict_to_json(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_config_from_json(filepath: str):
    with open(filepath, "r") as json_file:
        return json.load(json_file)


def grad_analysis(rank, world_size, debug=False, json_config="grad_analysis_cfg.json"):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    config_data = load_config_from_json(json_config)

    training_config = TrainingConfig(**config_data["training"])
    dataloader_config = DataloaderConfig(**config_data["dataloader"])
    model_config = ModelConfig(**config_data["model"])

    os.makedirs(training_config.save_folder, exist_ok=True)
    # paste the training config for this run
    dump_dict_to_json(
        config_data, f"{training_config.save_folder}/training_config.json"
    )

    # global training RNG
    torch.manual_seed(training_config.master_seed)
    random.seed(training_config.master_seed)

    # load model
    with torch.no_grad():
        # load chroma and enable grad
        chroma_params._use_compiled = True
        with torch.device("meta"):
            model = Chroma(chroma_params)
        model.load_state_dict(load_safetensors(model_config.chroma_path), assign=True)

        # randomly train inner layers at a time
        trained_double_blocks = list(range(len(model.double_blocks)))
        trained_single_blocks = list(range(len(model.single_blocks)))
        random.shuffle(trained_double_blocks)
        random.shuffle(trained_single_blocks)
        # lazy :P
        trained_double_blocks = trained_double_blocks * 1000000
        trained_single_blocks = trained_single_blocks * 1000000

        # load ae
        with torch.device("meta"):
            ae = AutoEncoder(ae_params)
        ae.load_state_dict(load_safetensors(model_config.vae_path), assign=True)
        ae.to(torch.bfloat16)

        # load t5
        t5_tokenizer = T5Tokenizer.from_pretrained(model_config.t5_tokenizer_path)
        t5_config = T5Config.from_json_file(model_config.t5_config_path)
        with torch.device("meta"):
            t5 = T5EncoderModel(t5_config)
        t5.load_state_dict(
            replace_keys(load_file_multipart(model_config.t5_path)), assign=True
        )
        t5.eval()
        t5.to(torch.bfloat16)
        if model_config.t5_to_8bit:
            cast_linear(t5, torch.float8_e4m3fn)

    dataset = TextImageDataset(
        batch_size=dataloader_config.batch_size,
        jsonl_path=dataloader_config.jsonl_metadata_path,
        image_folder_path=dataloader_config.image_folder_path,
        # don't use this tag implication pruning it's slow!
        # preprocess the jsonl tags before training!
        # tag_implication_path="tag_implications.csv",
        base_res=dataloader_config.base_resolution,
        shuffle_tags=dataloader_config.shuffle_tags,
        tag_drop_percentage=dataloader_config.tag_drop_percentage,
        uncond_percentage=dataloader_config.uncond_percentage,
        resolution_step=dataloader_config.resolution_step,
        seed=training_config.master_seed,
        rank=rank,
        num_gpus=world_size,
        ratio_cutoff=dataloader_config.ratio_cutoff,
    )

    hooks = []
    optimizer_counter = 0
    negative_embed = None

    global_step = training_config.aim_steps
    while True:
        training_config.master_seed += 1
        torch.manual_seed(training_config.master_seed)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # batch size is handled in the dataset
            shuffle=False,
            num_workers=dataloader_config.num_workers,
            prefetch_factor=dataloader_config.prefetch_factor,
            pin_memory=True,
            collate_fn=dataset.dummy_collate_fn,
        )
        for counter, data in tqdm(
            enumerate(dataloader),
            total=len(dataset),
            desc=f"training, Rank {rank}",
            position=rank,
        ):
            images, caption, index, loss_weighting = data[0]
            # just in case the dataloader is failing
            caption = [x if x is not None else "" for x in caption]
            caption = [x.lower() if torch.rand(1).item() < 0.25 else x for x in caption]
            # flag the uncond as 0 for cfg training
            is_uncond_onehot = one_hot_presence_tensor(caption).to(rank, non_blocking=True) * training_config.contrastive_strength

            loss_weighting = torch.tensor(loss_weighting, device=rank)
            if counter % training_config.change_layer_every == 0:
                # periodically remove the optimizer and swap it with new one

                # aliasing to make it cleaner
                o_c = optimizer_counter
                n_ls = training_config.trained_single_blocks
                n_ld = training_config.trained_double_blocks
                trained_layer_keywords = (
                    [
                        f"double_blocks.{x}."
                        for x in trained_double_blocks[o_c * n_ld : o_c * n_ld + n_ld]
                    ]
                    + [
                        f"single_blocks.{x}."
                        for x in trained_single_blocks[o_c * n_ls : o_c * n_ls + n_ls]
                    ]
                    + ["txt_in", "img_in", "final_layer"]
                )

                # remove hooks and load the new hooks
                if len(hooks) != 0:
                    hooks = [hook.remove() for hook in hooks]

                _, _, hooks, trained_params = init_optimizer(
                    model,
                    trained_layer_keywords,
                    training_config.lr,
                    training_config.weight_decay,
                    training_config.warmup_steps,
                )

                optimizer_counter += 1

            # we load and unload vae and t5 here to reduce vram usage
            # think of this as caching on the fly
            # load t5 and vae to GPU
            ae.to(rank)
            t5.to(rank)

            acc_latents = []
            acc_embeddings = []
            acc_mask = []
            acc_embeddings_neg = []
            acc_mask_neg = []
            for mb_i in tqdm(
                range(
                    dataloader_config.batch_size
                    // training_config.cache_minibatch
                    // world_size
                ),
                desc=f"preparing latents, Rank {rank}",
                position=rank,
            ):
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    # init random noise
                    text_inputs = t5_tokenizer(
                        caption[
                            mb_i
                            * training_config.cache_minibatch : mb_i
                            * training_config.cache_minibatch
                            + training_config.cache_minibatch
                        ],
                        padding="max_length",
                        max_length=model_config.t5_max_length,
                        truncation=True,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_tensors="pt",
                    ).to(t5.device)

                    # offload to cpu
                    t5_embed = t5(text_inputs.input_ids, text_inputs.attention_mask).to(
                        "cpu", non_blocking=True
                    )
                    acc_embeddings.append(t5_embed)
                    acc_mask.append(text_inputs.attention_mask)

                    # flush
                    torch.cuda.empty_cache()
                    latents = ae.encode_for_train(
                        images[
                            mb_i
                            * training_config.cache_minibatch : mb_i
                            * training_config.cache_minibatch
                            + training_config.cache_minibatch
                        ].to(rank)
                    ).to("cpu", non_blocking=True)
                    acc_latents.append(latents)

                    # flush
                    torch.cuda.empty_cache()

                    # only do this one time to compute negative embeddings
                    if negative_embed == None:
                        text_inputs_neg = t5_tokenizer(
                            [""] * training_config.cache_minibatch,
                            padding="max_length",
                            max_length=model_config.t5_max_length,
                            truncation=True,
                            return_length=False,
                            return_overflowing_tokens=False,
                            return_tensors="pt",
                        ).to(t5.device)

                        # offload to cpu
                        t5_embed_neg = t5(text_inputs_neg.input_ids, text_inputs_neg.attention_mask).to(
                            "cpu", non_blocking=True
                        )
                        acc_embeddings_neg.append(t5_embed_neg)
                        acc_mask_neg.append(text_inputs_neg.attention_mask)
            

            # accumulate the latents and embedding in a variable
            # unload t5 and vae

            t5.to("cpu")
            ae.to("cpu")
            torch.cuda.empty_cache()
            if not debug:
                dist.barrier()

            # move model to device
            model.to(rank)

            acc_latents = torch.cat(acc_latents, dim=0)
            acc_embeddings = torch.cat(acc_embeddings, dim=0)
            acc_mask = torch.cat(acc_mask, dim=0)

            # only do this one time to compute negative embeddings
            if negative_embed == None:
                negative_embed = (torch.cat(acc_embeddings_neg, dim=0), torch.cat(acc_mask_neg, dim=0))

            # process the cache buffer now!
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                # prepare flat image and the target lerp
                (
                    noisy_latents,
                    target,
                    input_timestep,
                    image_pos_id,
                    latent_shape,
                ) = prepare_sot_pairings(acc_latents.to(rank))
                noisy_latents = noisy_latents.to(torch.bfloat16)
                target = target.to(torch.bfloat16)
                input_timestep = input_timestep.to(torch.bfloat16)
                image_pos_id = image_pos_id.to(rank)

                # t5 text id for the model
                text_ids = torch.zeros((noisy_latents.shape[0], 512, 3), device=rank)
                # NOTE:
                # using static guidance 1 for now
                # this should be disabled later on !
                static_guidance = torch.tensor(
                    [0.0] * acc_latents.shape[0], device=rank
                )

            # set the input to requires grad to make autograd works
            noisy_latents.requires_grad_(True)
            acc_embeddings.requires_grad_(True)

            ot_bs = acc_latents.shape[0]

            # aliasing
            mb = training_config.train_minibatch
            loss_log = []
            for tmb_i in tqdm(
                range(dataloader_config.batch_size // mb // world_size),
                desc=f"minibatch training, Rank {rank}",
                position=rank,
            ):
                # CFG distilled target
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        acc_embeddings_neg, acc_mask_neg = negative_embed
                        pred_neg = model(
                            img=noisy_latents[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            img_ids=image_pos_id[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            txt=acc_embeddings_neg[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            txt_ids=text_ids[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            txt_mask=acc_mask_neg[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            timesteps=input_timestep[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                            guidance=static_guidance[tmb_i * mb : tmb_i * mb + mb].to(
                                rank, non_blocking=True
                            ),
                        )

                    # target vector for training

                    # flags to mark which one is uncond
                    # reshape to [batch_size, 1, 1, ...]
                    flags = is_uncond_onehot[tmb_i * mb : tmb_i * mb + mb].view(-1, *([1] * (target.ndim - 1)))
                    # set all to have CFG
                    target_cfg_vector = pred_neg + training_config.contrastive_strength * (target[tmb_i * mb : tmb_i * mb + mb] - pred_neg)
                    # mask and replace the uncond with classical target
                    target_cfg_vector = torch.where(flags.bool(), target_cfg_vector, target[tmb_i * mb : tmb_i * mb + mb])

                # do this inside for loops!
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(
                        img=noisy_latents[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        img_ids=image_pos_id[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt=acc_embeddings[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt_ids=text_ids[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt_mask=acc_mask[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        timesteps=input_timestep[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        guidance=static_guidance[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                    )
                    # TODO: need to scale the loss with rank count and grad accum!

                    # Compute per-element squared error and mean over sequence and feature dims
                    loss = ((pred - target_cfg_vector) ** 2).mean(dim=(1, 2))  # Shape: [mb]

                    # Normalize per full batch
                    loss = loss / (dataloader_config.batch_size // mb)  # Shape: [mb]

                    # Apply per-sample weight
                    weights = loss_weighting[tmb_i * mb : tmb_i * mb + mb]  # Shape: [mb]

                    # Normalize weights to ensure the overall loss scale is consistent
                    weights = weights / weights.sum()

                    # Compute final weighted loss
                    loss = (loss * weights).sum()

                    # correct!
                    # loss = F.mse_loss(
                    #     pred,
                    #     target[tmb_i * mb : tmb_i * mb + mb],
                    # ) / (dataloader_config.batch_size // mb)
                torch.cuda.empty_cache()
                loss.backward()
                loss_log.append(
                    loss.detach().clone() * (dataloader_config.batch_size // mb)
                )
            loss_log = sum(loss_log) / len(loss_log)
            # offload some params to cpu just enough to make room for the caching process
            # and only offload non trainable params
            del acc_embeddings, noisy_latents, acc_latents
            torch.cuda.empty_cache()

            StochasticAccumulator.reassign_grad_buffer(model)

            if not debug:
                synchronize_gradients(model)

            torch.cuda.empty_cache()

            if rank == 0:
                model_filename = f"{training_config.save_folder}/{global_step}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.grad"

                gradients = {}
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        gradients[n] = p.grad
                    else:
                        gradients[n] = torch.zeros_like(p)
                torch.save(
                    gradients,
                    model_filename,
                )
                if training_config.hf_token:
                    upload_to_hf(
                        model_filename,
                        model_filename,
                        training_config.hf_repo_id,
                        training_config.hf_token,
                    )
            if not debug:
                dist.barrier()


            # flush
            acc_latents = []
            acc_embeddings = []
            global_step += 1

            break # only 1 training loop needed
        break # only 1 training loop needed


    if not debug:
        dist.destroy_process_group()

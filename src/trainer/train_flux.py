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

from transformers import T5Tokenizer, CLIPTokenizer, CLIPTextModel
import wandb

from src.dataloaders.dataloader import TextImageDataset
from src.models.flux.model import Flux, flux_params
from src.models.flux.sampling import get_noise, get_schedule, denoise
from src.models.flux.utils import (
    vae_flatten,
    prepare_latent_image_ids,
    vae_unflatten,
    calculate_shift,
    time_shift,
)
from src.models.flux.module.autoencoder import AutoEncoder, ae_params
from src.math_utils import cosine_optimal_transport
from src.models.flux.module.t5 import T5EncoderModel, T5Config, replace_keys
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
    clip_max_length: int


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

    flux_path: str
    vae_path: str
    t5_path: str
    t5_config_path: str
    t5_tokenizer_path: str
    t5_to_8bit: bool
    t5_max_length: int
    clip_path: str
    clip_tokenizer_path: str
    clip_max_length: int


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
    # init optimizer
    optimizer = Compass(
        [
            {
                "params": [
                    param
                    for name, param in trained_params
                    if ("bias" not in name and "norm" not in name)
                ]
            },
            {
                "params": [
                    param
                    for name, param in trained_params
                    if ("bias" in name or "norm" in name)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    return optimizer, scheduler, hooks, trained_params



def synchronize_gradients(model, scale=1):
    for param in model.parameters():
        if param.grad is not None:
            # Synchronize gradients by summing across all processes
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Average the gradients if needed
            if scale > 1:
                param.grad /= scale


def optimizer_state_to(optimizer, device):
    for param, state in optimizer.state.items():
        for key, value in state.items():
            # Check if the item is a tensor
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=True)



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


def inference_wrapper(
    model,
    ae,
    clip_tokenizer,
    clip,
    t5_tokenizer,
    t5,
    seed: int,
    steps: int,
    guidance: int,
    prompts: list,
    rank: int,
    image_dim=(512, 512),
    t5_max_length=512,
    clip_max_length=77,
):
    
    DEVICE = model.device
    PROMPT = prompts
    T5_MAX_LENGTH = t5_max_length
    CLIP_MAX_LENGTH = clip_max_length

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise = get_noise(len(PROMPT), image_dim[1], image_dim[0], DEVICE, torch.bfloat16, seed)
            noise, shape = vae_flatten(noise)
            noise = noise.to(rank)
            n, c, h, w = shape
            image_pos_id = prepare_latent_image_ids(n, h, w).to(rank)

            timesteps = get_schedule(steps, noise.shape[1])

            model.to("cpu")
            ae.to("cpu")
            clip.to("cpu")
            t5.to(rank)
            text_inputs = t5_tokenizer(
                PROMPT,
                padding="max_length",
                max_length=T5_MAX_LENGTH,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).to(t5.device)
            t5_embed = t5(text_inputs.input_ids, text_inputs.attention_mask).to(rank)
            t5.to("cpu")

            clip.to(rank)
            clip_text_inputs = clip_tokenizer(
                PROMPT,
                truncation=True,
                max_length=CLIP_MAX_LENGTH,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            ).to(clip.device)
            pooled_clip_embed = clip(
                input_ids=clip_text_inputs.input_ids,
                attention_mask=None,
                output_hidden_states=False,
            ).pooler_output.to(rank)
            clip.to("cpu")

            text_ids = torch.zeros((len(PROMPT), T5_MAX_LENGTH, 3), device=rank)

            model.to(rank)
            latent = denoise(
                model,
                noise,
                image_pos_id,
                t5_embed,
                text_ids,
                text_inputs.attention_mask,
                pooled_clip_embed,
                timesteps,
                guidance,
            )
            model.to("cpu")
            
            ae.to(rank)
            output_image = ae.decode(vae_unflatten(latent, shape))
            ae.to("cpu")

    return output_image


def train_flux(rank, world_size, debug=False):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    config_data = load_config_from_json("training_config.json")

    training_config = TrainingConfig(**config_data["training"])
    inference_config = InferenceConfig(**config_data["inference"])
    dataloader_config = DataloaderConfig(**config_data["dataloader"])
    model_config = ModelConfig(**config_data["model"])
    extra_inference_config = [
        InferenceConfig(**conf) for conf in config_data["extra_inference_config"]
    ]


    # Setup Aim run
    if training_config.aim_path is not None and rank == 0:
        # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = Run(repo=training_config.aim_path, run_hash=training_config.aim_hash, experiment=training_config.aim_experiment_name)

        hparams = config_data.copy()
        hparams["training"]['aim_path'] = None
        run["hparams"] = hparams


    os.makedirs(training_config.save_folder, exist_ok=True)
    # paste the training config for this run
    dump_dict_to_json(
        config_data, f"{training_config.save_folder}/training_config.json"
    )
    os.makedirs(inference_config.inference_folder, exist_ok=True)
    
    torch.manual_seed(training_config.master_seed)
    random.seed(training_config.master_seed)

    with torch.no_grad():
        # load chroma and enable grad
        flux_params._use_compiled = True
        with torch.device("meta"):
            model = Flux(flux_params)
        model.load_state_dict(load_safetensors(model_config.flux_path), assign=True)

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

        clip_tokenizer = CLIPTokenizer.from_pretrained(model_config.clip_tokenizer_path)
        clip = CLIPTextModel.from_pretrained(model_config.clip_path).eval().requires_grad_(False)
        clip.to(torch.bfloat16)


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

    optimizer = None
    scheduler = None
    hooks = []
    optimizer_counter = 0

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
                    + ["txt_in", "img_in", "final_layer", "vector_in"]
                )

                # remove hooks and load the new hooks
                if len(hooks) != 0:
                    hooks = [hook.remove() for hook in hooks]

                optimizer, scheduler, hooks, trained_params = init_optimizer(
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
            clip.to(rank)

            acc_latents = []
            acc_t5_embeddings = []
            acc_clip_embeddings = []
            acc_mask = [] 
            num_cache_iters = dataloader_config.batch_size // training_config.cache_minibatch // world_size
            for mb_i in tqdm(range(num_cache_iters), desc=f"preparing latents, Rank {rank}", position=rank):
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    start_idx = mb_i * training_config.cache_minibatch
                    end_idx = start_idx + training_config.cache_minibatch
                    
                    mini_batch_captions = caption[start_idx:end_idx]
                    
                    # T5 embeddings
                    text_inputs = t5_tokenizer(
                        mini_batch_captions, padding="max_length", max_length=model_config.t5_max_length,
                        truncation=True, return_tensors="pt"
                    ).to(t5.device)
                    t5_embed = t5(text_inputs.input_ids, text_inputs.attention_mask).to("cpu", non_blocking=True)
                    acc_t5_embeddings.append(t5_embed)
                    acc_mask.append(text_inputs.attention_mask.to("cpu", non_blocking=True))
                    
                    # CLIP embeddings
                    clip_inputs = clip_tokenizer(
                        mini_batch_captions, padding="max_length", max_length=model_config.clip_max_length,
                        truncation=True, return_tensors="pt"
                    ).to(clip.device)
                    clip_embed = clip(input_ids=clip_inputs.input_ids).pooler_output.to("cpu", non_blocking=True)
                    acc_clip_embeddings.append(clip_embed)

                    # VAE latents
                    latents = ae.encode_for_train(images[start_idx:end_idx].to(rank)).to("cpu", non_blocking=True)
                    acc_latents.append(latents)
                    torch.cuda.empty_cache()

            # accumulate the latents and embedding in a variable
            # unload t5 and vae

            t5.to("cpu")
            ae.to("cpu")
            clip.to("cpu")
            torch.cuda.empty_cache()
            if not debug:
                dist.barrier()

            model.to(rank)

            acc_latents = torch.cat(acc_latents, dim=0)
            acc_t5_embeddings = torch.cat(acc_t5_embeddings, dim=0)
            acc_clip_embeddings = torch.cat(acc_clip_embeddings, dim=0)
            acc_mask = torch.cat(acc_mask, dim=0)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                noisy_latents, target, input_timestep, image_pos_id, _ = prepare_sot_pairings(acc_latents.to(rank))
                noisy_latents, target, input_timestep = noisy_latents.bfloat16(), target.bfloat16(), input_timestep.bfloat16()
                image_pos_id = image_pos_id.to(rank)
                text_ids = torch.zeros((noisy_latents.shape[0], model_config.t5_max_length, 3), device=rank)
                static_guidance = torch.zeros(acc_latents.shape[0], device=rank) #TODO change this to online training

            mb = training_config.train_minibatch
            loss_log = []
            
            num_train_iters = dataloader_config.batch_size // mb // world_size
            for tmb_i in tqdm(range(num_train_iters), desc=f"minibatch training, Rank {rank}", position=rank):
                start_idx, end_idx = tmb_i * mb, (tmb_i + 1) * mb
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(
                        img=noisy_latents[start_idx:end_idx].to(rank, non_blocking=True),
                        img_ids=image_pos_id[start_idx:end_idx].to(rank, non_blocking=True),
                        txt=acc_t5_embeddings[start_idx:end_idx].to(rank, non_blocking=True),
                        txt_ids=text_ids[start_idx:end_idx].to(rank, non_blocking=True),
                        txt_mask=acc_mask[start_idx:end_idx].to(rank, non_blocking=True),
                        y=acc_clip_embeddings[start_idx:end_idx].to(rank, non_blocking=True),
                        timesteps=input_timestep[start_idx:end_idx].to(rank, non_blocking=True),
                        guidance=static_guidance[start_idx:end_idx].to(rank, non_blocking=True),
                    )
                    
                    loss = ((pred - target[start_idx:end_idx]) ** 2).mean(dim=(1, 2))
                    loss = loss / num_train_iters
                    weights = loss_weighting[start_idx:end_idx] / loss_weighting[start_idx:end_idx].sum()
                    loss = (loss * weights).sum()

                loss.backward()
                loss_log.append(loss.detach().clone() * num_train_iters)
            
            loss_log = sum(loss_log) / len(loss_log) if loss_log else torch.tensor(0.0)
            
            del acc_t5_embeddings, noisy_latents, acc_latents, acc_clip_embeddings
            torch.cuda.empty_cache()

            offload_param_count = 0
            for name, param in model.named_parameters():
                if not param.requires_grad and offload_param_count < training_config.offload_param_count:
                    offload_param_count += param.numel()
                    param.data = param.data.to("cpu", non_blocking=True)

            optimizer_state_to(optimizer, rank)
            StochasticAccumulator.reassign_grad_buffer(model)

            if not debug:
                synchronize_gradients(model)

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                run.track(loss_log, name='loss', step=global_step)
                run.track(optimizer.param_groups[0]['lr'], name='learning_rate', step=global_step)

            optimizer_state_to(optimizer, "cpu")
            torch.cuda.empty_cache()

            if (counter + 1) % training_config.save_every == 0 and rank == 0:
                model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
                torch.save(model.state_dict(), model_filename)
                if training_config.hf_token:
                    upload_to_hf(model_filename, model_filename, training_config.hf_repo_id, training_config.hf_token)

            if (counter + 1) % inference_config.inference_every == 0:
                all_grids = []
                preview_prompts = inference_config.prompts + caption[:1]

                inference_configs = [inference_config] + extra_inference_config
                for conf in inference_configs:
                    for prompt in preview_prompts:
                        images_tensor = inference_wrapper(
                            model=model, ae=ae, clip_tokenizer=clip_tokenizer, clip=clip, t5_tokenizer=t5_tokenizer, t5=t5,
                            seed=training_config.master_seed + rank, steps=conf.steps, guidance=conf.guidance,
                            prompts=[prompt], rank=rank, image_dim=conf.image_dim,
                            t5_max_length=conf.t5_max_length, clip_max_length=conf.clip_max_length
                        )

                        if not debug:
                            gather_list = [torch.empty_like(images_tensor) for _ in range(world_size)] if rank == 0 else None
                            dist.gather(images_tensor, gather_list=gather_list, dst=0)
                            if rank == 0:
                                gathered_images = torch.cat(gather_list, dim=0)
                        else:
                            gathered_images = images_tensor

                        if rank == 0:
                            grid = make_grid(gathered_images.clamp(-1, 1).add(1).div(2), nrow=8, normalize=True)
                            all_grids.append(grid)
                
                all_prompt_parts = [[] for _ in range(world_size)]
                if not debug:
                    dist.all_gather_object(all_prompt_parts, caption[:1])
                
                if rank == 0:
                    all_prompts = [p for sublist in all_prompt_parts for p in sublist]
                    final_grid = torch.cat(all_grids, dim=1)
                    file_path = os.path.join(inference_config.inference_folder, f"{counter}.jpg")
                    save_image(final_grid, file_path)
                    
                    img_pil = PILImage.open(file_path)
                    aim_img = AimImage(img_pil, caption="\n".join(preview_prompts + all_prompts))
                    run.track(aim_img, name='example_image', step=global_step)

            global_step += 1

        if rank == 0:
            model_filename = f"{training_config.save_folder}/final_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
            torch.save(model.state_dict(), model_filename)
            if training_config.hf_token:
                upload_to_hf(model_filename, model_filename, training_config.hf_repo_id, training_config.hf_token)

    if not debug:
        dist.destroy_process_group()
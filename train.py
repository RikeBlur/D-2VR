#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#2025.02.06 ljf

import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from omegaconf import OmegaConf
import cv2
import json
from reds_dataset import REDSDataset
from degradation.apply_degradation import apply_dynamic_degradation_batch
from eval import init_eval_metrics, compute_metrics

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig    
from einops import rearrange
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from opticalflow.flow_utils import get_flow, flow_warp

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel
)

from pipeline.d2vr_pipeline import D2VRPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from scheduler.ddpm_scheduler import DDPMScheduler

# load disc and target model
from core.build import build_disc, build_target_model
from core.optimizer import build_opt
from torch.utils.tensorboard import SummaryWriter

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def run_validation(vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler, val_dataloader, 
                   accelerator, weight_dtype, of_model, step, eval_metrics=None, log_file=None):
    """Run validation and compute evaluation scores on the validation set."""
    logger.info("Running validation... ")
    
    controlnet.eval()
    unet.eval()
    
    if eval_metrics is None:
        eval_metrics = init_eval_metrics(device=accelerator.device)
    
    metric_sums = {
        'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'dists': 0.0,
        'musiq': 0.0, 'clip': 0.0, 
        'tlpips': 0.0, 'tof': 0.0
    }
    tlpips_count = 0
    tof_count = 0
    num_samples = 0
    
    with torch.no_grad():
        tokenization = tokenizer([''] * 1, max_length=tokenizer.model_max_length, 
                                padding="max_length", truncation=True, return_tensors="pt")
        encoder_hidden_states_single = text_encoder(tokenization.input_ids.to(accelerator.device))[0]
    
    with torch.no_grad():
        for batch in val_dataloader:
            lq = batch['lq'].to(accelerator.device)
            gt = batch['gt'].to(accelerator.device)
            gt = 2 * gt - 1
            # lq = 2 * lq - 1
            lq, _ = apply_dynamic_degradation_batch(gt, gt.device, args.degradation_params_json)
            b, t, _, _, _ = lq.shape

            
            
            upscaled_lq = rearrange(lq, 'b t c h w -> (b t) c h w')
            upscaled_lq = F.interpolate(upscaled_lq, scale_factor=4, mode='bicubic')
            upscaled_lq = rearrange(upscaled_lq, '(b t) c h w -> b t c h w', b=b, t=t)
            
            encoder_hidden_states = encoder_hidden_states_single.repeat(b, 1, 1)
            
            random_t = [round(random.random()) * 2 for _ in range(b)]
            gt_prev = torch.stack([gt[i, frame_idx] for i, frame_idx in enumerate(random_t)])
            upscaled_lq_prev = torch.stack([upscaled_lq[i, frame_idx] for i, frame_idx in enumerate(random_t)])
            lq_prev = torch.stack([lq[i, frame_idx] for i, frame_idx in enumerate(random_t)])
            
            gt_cur = gt[:, t // 2, ...]
            lq_cur = lq[:, t // 2, ...]
            upscaled_lq_cur = upscaled_lq[:, t // 2, ...]
            
            # Convert to latent space
            latents_prev = vae.encode(gt_prev.to(dtype=weight_dtype)).latent_dist.sample()
            latents_prev = latents_prev * vae.config.scaling_factor
            
            # Add noise
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=latents_prev.device).long()
            
            noise_prev = torch.randn_like(latents_prev)
            noisy_latents_prev = noise_scheduler.add_noise(latents_prev, noise_prev, timesteps)
            noisy_latents_prev_cat = torch.cat([noisy_latents_prev, lq_prev], dim=1)
            
            # Previous frame prediction
            model_pred_prev = unet(
                noisy_latents_prev_cat,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            approximated_x0_latent_prev = noise_scheduler.get_approximated_x0(model_pred_prev, timesteps, noisy_latents_prev)
            approximated_x0_rgb_prev = vae.decode(approximated_x0_latent_prev / vae.config.scaling_factor).sample
            
            # Warp
            f_flow = get_flow(of_model, upscaled_lq_cur, upscaled_lq_prev)
            warped_approximated_x0 = flow_warp(approximated_x0_rgb_prev, f_flow)
            controlnet_image = warped_approximated_x0.to(dtype=weight_dtype)
            
            # Current frame - generate reconstruction
            latents_cur = vae.encode(gt_cur.to(dtype=weight_dtype)).latent_dist.sample()
            latents_cur = latents_cur * vae.config.scaling_factor
            noise_cur = torch.randn_like(latents_cur)
            noisy_latents_cur = noise_scheduler.add_noise(latents_cur, noise_cur, timesteps)
            noisy_latents_cur_cat = torch.cat([noisy_latents_cur, lq_cur], dim=1)
            
            # ControlNet
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents_cur_cat,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )
            
            # UNet prediction
            model_pred = unet(
                noisy_latents_cur_cat,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample
            
            approximated_x0_latent = noise_scheduler.get_approximated_x0(model_pred, timesteps, noisy_latents_cur)
            rec = vae.decode(approximated_x0_latent / vae.config.scaling_factor).sample
            
            gt_cur_01 = ((gt_cur + 1.0) / 2.0).clamp(0, 1)
            rec_01 = ((rec + 1.0) / 2.0).clamp(0, 1)
            gt_prev_01 = ((gt_prev + 1.0) / 2.0).clamp(0, 1)
            rec_prev_01 = ((approximated_x0_rgb_prev + 1.0) / 2.0).clamp(0, 1)
            
            for i in range(b):
                results = compute_metrics(
                    gt_cur_01[i], rec_01[i],
                    prev_gt=gt_prev_01[i], prev_rec=rec_prev_01[i],
                    metrics=eval_metrics,
                    of_model=of_model
                )
                
                metric_sums['psnr'] += results['psnr']
                metric_sums['ssim'] += results['ssim']
                metric_sums['lpips'] += results['lpips']
                metric_sums['dists'] += results['dists']
                metric_sums['musiq'] += results['musiq']
                metric_sums['clip'] += results['clip']
                
                if 'tlpips' in results:
                    metric_sums['tlpips'] += results['tlpips']
                    tlpips_count += 1
                if 'tof' in results:
                    metric_sums['tof'] += results['tof']
                    tof_count += 1
                
                num_samples += 1
    
    avg_metrics = {}
    for key in ['psnr', 'ssim', 'lpips', 'dists', 'musiq', 'clip']:
        avg_metrics[key] = metric_sums[key] / num_samples if num_samples > 0 else 0.0
    
    avg_metrics['tlpips'] = metric_sums['tlpips'] / tlpips_count if tlpips_count > 0 else 0.0
    avg_metrics['tof'] = metric_sums['tof'] / tof_count if tof_count > 0 else 0.0
    
    val_score = avg_metrics['psnr'] + avg_metrics['ssim'] * 10 - avg_metrics['lpips'] * 10 - avg_metrics['dists'] * 10
    
    logger.info(f"Validation - Step {step}:")
    logger.info(f"  PSNR: {avg_metrics['psnr']:.3f}, SSIM: {avg_metrics['ssim']:.4f}, "
                f"LPIPS: {avg_metrics['lpips']:.4f}, DISTS: {avg_metrics['dists']:.4f}")
    logger.info(f"  MUSIQ: {avg_metrics['musiq']:.3f}, CLIP: {avg_metrics['clip']:.4f}")
    logger.info(f"  tLPIPS: {avg_metrics['tlpips']*1e3:.3f}, tOF: {avg_metrics['tof']*1e1:.3f}")
    logger.info(f"  Validation Score: {val_score:.4f}")
    
    if log_file is not None and accelerator.is_main_process:
        log_entry = {
            'step': step,
            'val_score': float(val_score),
            'metrics': {k: float(v) for k, v in avg_metrics.items()}
        }
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = []
        else:
            log_data = []
        
        log_data.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Validation results saved to {log_file}")
    
    controlnet.train()
    unet.train()
    
    return val_score, avg_metrics, eval_metrics


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):

    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-2-1',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-2-1',
        required=True,
        help="Path to pretrained vae model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or model identifier from huggingface.co/models."
        " If not specified unet will be loaded from pretrained_model_name_or_path.",
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        help="Whether to train the unet model.",
    )
    parser.add_argument(
        "--train_controlnet",
        action="store_true",
        default=True,
        help="Whether to train the controlnet model. Default is True.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--degradation_params_json",
        type=str,
        default=None,
        help="Path to degradation parameters JSON file.",
    )
    parser.add_argument(
        "--esrgan",
        action="store_true",
        help="Whether to use esrgan degradation.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--of_model", type=str, default="DRFA", choices=["RAFT", "GMFlow", "FlowFormerPP", "GMA", "DRFA"], help="Optical flow model to use")
    parser.add_argument("--of_path", type=str, default="./DRFA/", help="Path to the optical flow model source code directory (required for GMFlow/FlowFormerPP/GMA/SC)")
    parser.add_argument("--of_pretrained", type=str, default=None, help="Path to the optical flow model pretrained weights")
    parser.add_argument("--adv_weight", type=float, default=0.01, help="Weight for the adversarial loss.")
    parser.add_argument("--tlpips_weight", type=float, default=0.1, help="Weight for the tLPIPS loss.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    #parser.add_argument(
    #    "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    #)
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        default=None,
        help=(
            "The path to the config file related to the dataset."
        ),
    )    
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        '--multiscale_D',
        action='store_true',
        help='Whether to use multi-scale Discriminator. Extra heads from intermediate features',
    )
    parser.add_argument(
        '--disc_model_path',
        type=str,
        default=None,
        help='Path to the pretrained SD model used to initialize the UNet discriminator backbone.',
    )
    parser.add_argument(
        '--G_lr',
        type=float,
        default=1e-6,
        help='learning rate for generator',
    )
    parser.add_argument(
        '--D_lr',
        type=float,
        default=1e-6,
        help='learning rate for discriminator',
    )
    parser.add_argument('--optimizer', type=str, default='adamw', 
                            choices=['adam', 'adamw', 'adafactor'],
                            help='Choices for optimizer, choose from adam, adamw, and adafactor')
    parser.add_argument(
        '--D_ts',
        type=str,
        default='0-750',
        help="""Timestep choices for Discriminator. Support two formats:
                choice1: 10, 249, 499, 749 for discrete values.
                choice2: 0-11, 200-250, 400-500, 700-750 for ranges.
                default is range 0-750"""
    )
    parser.add_argument(
        '--misaligned_pairs_D',
        action='store_true',
        help='Whether to use mis aligned pairs for Discriminator. Pair some real images with misaligned prompts to enforce text-image alignment abilities of Discriminator.',
    )
    parser.add_argument(
        '--num_ts',
        type=int,
        default=4,
        help=('Number of time steps to sample from the original 1000 time steps for training G. We train one-step model by default')
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None and args.dataset_config_path is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir` or `dataset_config_path`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    writer = SummaryWriter(log_dir=logging_dir) if accelerator.is_local_main_process else None

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # load disc and target model
    disc = build_disc(args.disc_model_path, args.multiscale_D)
    # target_model = build_target_model(args.pretrained_model_name_or_path)

    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path, subfolder="vae", revision=args.revision)
    
    if args.unet_model_name_or_path:
        logger.info("Loading existing unet weights")
        unet = UNet2DConditionModel.from_pretrained(args.unet_model_name_or_path)
    else:
        logger.info("Initializing unet with random weights")
        unet_config = UNet2DConditionModel.load_config(args.pretrained_model_name_or_path, subfolder="unet")
        unet = UNet2DConditionModel.from_config(unet_config)

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(64,128,256,)) 

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.train_controlnet:
                    controlnet_to_save = accelerator.unwrap_model(controlnet)
                    controlnet_folder = os.path.join(output_dir, "controlnet")
                    os.makedirs(controlnet_folder, exist_ok=True)
                    
                    logger.info(f"💾 Saving ControlNet to: {controlnet_folder}")
                    
                    config_path = os.path.join(controlnet_folder, "config.json")
                    if os.path.exists(config_path):
                        try:
                            os.remove(config_path)
                        except OSError:
                            pass

                    controlnet_to_save.save_pretrained(controlnet_folder)
                    
                    if not os.path.exists(os.path.join(controlnet_folder, "config.json")):
                        logger.warning("⚠️ config.json not found after save_pretrained, saving manually...")
                        controlnet_to_save.save_config(controlnet_folder)

                    logger.info("✅ ControlNet saved successfully.")
                
                if args.train_unet:
                    unet_to_save = accelerator.unwrap_model(unet)
                    unet_folder = os.path.join(output_dir, "unet")
                    os.makedirs(unet_folder, exist_ok=True)
                    
                    logger.info(f"💾 Saving UNet to: {unet_folder}")
                    
                    unet_config_path = os.path.join(unet_folder, "config.json")
                    if os.path.exists(unet_config_path):
                        try:
                            os.remove(unet_config_path)
                        except OSError:
                            pass
                    
                    unet_to_save.save_pretrained(unet_folder)
                    
                    if not os.path.exists(os.path.join(unet_folder, "config.json")):
                        logger.warning("⚠️ UNet config.json not found after save_pretrained, saving manually...")
                        unet_to_save.save_config(unet_folder)
                    
                    logger.info("✅ UNet saved successfully.")

            if len(weights) > 0:
                logger.info(f"⚠️ Dropping {len(weights)} items from automatic accelerator save to prevent corruption.")
            
            while len(weights) > 0:
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.train_controlnet:
                controlnet_load_path = os.path.join(input_dir, "controlnet")
                
                if not os.path.exists(controlnet_load_path):
                    logger.warning(f"ControlNet checkpoint not found at {controlnet_load_path}, skipping load.")
                else:
                    try:
                        load_model = ControlNetModel.from_pretrained(controlnet_load_path)
                        model_to_load = accelerator.unwrap_model(controlnet)
                        model_to_load.register_to_config(**load_model.config)
                        model_to_load.load_state_dict(load_model.state_dict())
                        del load_model
                        logger.info(f"✅ ControlNet loaded from {controlnet_load_path}")
                    except Exception as e:
                        logger.error(f"Failed to load ControlNet from {controlnet_load_path}: {e}")
            
            if args.train_unet:
                unet_load_path = os.path.join(input_dir, "unet")
                
                if not os.path.exists(unet_load_path):
                    logger.warning(f"UNet checkpoint not found at {unet_load_path}, skipping load.")
                else:
                    try:
                        load_unet = UNet2DConditionModel.from_pretrained(unet_load_path)
                        unet_to_load = accelerator.unwrap_model(unet)
                        unet_to_load.register_to_config(**load_unet.config)
                        unet_to_load.load_state_dict(load_unet.state_dict())
                        del load_unet
                        logger.info(f"✅ UNet loaded from {unet_load_path}")
                    except Exception as e:
                        logger.error(f"Failed to load UNet from {unet_load_path}: {e}")

            while len(models) > 0:
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    device = accelerator.device
    if args.of_model == "RAFT":
        of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        of_model.requires_grad_(False)
        of_model = of_model.to(device)
        print("Using optical flow model: RAFT")
    elif args.of_model == "GMFlow":
        try:
            from opticalflow.gmflow_wrapper import GMFlowWrapper
            import sys
            if args.of_path is None:
                raise ValueError("--of_path is required for GMFlow (path to gmflow source directory)")
            sys.path.insert(0, args.of_path)
            from gmflow.gmflow import GMFlow
            
            gmflow_model = GMFlow(
                feature_channels=128,
                num_scales=1,
                upsample_factor=8,
                num_head=1,
                attention_type='swin',
                ffn_dim_expansion=4,
                num_transformer_layers=6,
            )
            
            if args.of_pretrained is None:
                raise ValueError("--of_pretrained is required for GMFlow (path to pretrained weights)")
            checkpoint = torch.load(args.of_pretrained, map_location='cpu')
            if 'model' in checkpoint:
                gmflow_model.load_state_dict(checkpoint['model'], strict=True)
            else:
                gmflow_model.load_state_dict(checkpoint, strict=True)
            
            gmflow_model = gmflow_model.to(device)
            gmflow_model.eval()
            
            of_model = GMFlowWrapper(gmflow_model)
            of_model.requires_grad_(False)
            print("Using optical flow model: GMFlow")
        except Exception as e:
            print(f"Failed to load GMFlow: {e}")
            print("Falling back to RAFT model...")
            of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            of_model.requires_grad_(False)
            of_model = of_model.to(device)
            print("Using optical flow model: RAFT (fallback)")
    elif args.of_model == "FlowFormerPP":
        try:
            from opticalflow.flowformer_wrapper import FlowFormerPPWrapper
            import sys
            if args.of_path is None:
                raise ValueError("--of_path is required for FlowFormerPP (path to FlowFormerPlusPlus source directory)")
            sys.path.insert(0, args.of_path)
            try:
                from core.FlowFormer import build_flowformer
                from configs.things import get_cfg as ffpp_get_cfg
            except ImportError:
                raise

            cfg = ffpp_get_cfg()
            flowformer_model = build_flowformer(cfg)
            if args.of_pretrained is None:
                raise ValueError("--of_pretrained is required for FlowFormerPP (path to pretrained weights)")
            checkpoint = torch.load(args.of_pretrained, map_location='cpu')
            state = checkpoint['model'] if 'model' in checkpoint else checkpoint
            if all(k.startswith('module.') for k in state.keys()):
                state = {k[len('module.'):]: v for k, v in state.items()}
            missing, unexpected = flowformer_model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"FlowFormer++ weights not strictly matched, missing={len(missing)}, unexpected={len(unexpected)}")

            flowformer_model = flowformer_model.to(device)
            flowformer_model.eval()

            of_model = FlowFormerPPWrapper(flowformer_model)
            of_model.requires_grad_(False)
            print("Using optical flow model: FlowFormer++")
        except Exception as e:
            print(f"Failed to load FlowFormer++: {e}")
            print("Falling back to RAFT model...")
            of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            of_model.requires_grad_(False)
            of_model = of_model.to(device)
            print("Using optical flow model: RAFT (fallback)")
    elif args.of_model == "GMA":
        try:
            from opticalflow.gma_wrapper import GMAWrapper
            import sys
            
            if args.of_path is None:
                raise ValueError("--of_path is required for GMA (path to GMA source directory)")
            gma_path = args.of_path
            gma_core_path = os.path.join(gma_path, 'core')
            for p in [gma_path, gma_core_path]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            
            import core.update
            import gma
            from core.network import RAFTGMA
            
            class Args:
                def __init__(self):
                    self.mixed_precision = False
                    self.alternate_corr = False
                    self.dropout = 0.0
                    self.num_heads = 1
                    self.position_only = False
                    self.position_and_content = True
                    self.pe = 'linear'
                    self.self_pe = 'linear'
                    self.corr_levels = 4
                    self.corr_radius = 4
                def __contains__(self, key):
                    return hasattr(self, key)
            
            gma_args = Args()
            gma_model = RAFTGMA(gma_args)
            
            if args.of_pretrained is None:
                raise ValueError("--of_pretrained is required for GMA (path to pretrained weights)")
            checkpoint = torch.load(args.of_pretrained, map_location='cpu')
            if 'model' in checkpoint:
                state = checkpoint['model']
            else:
                state = checkpoint
            
            if all(k.startswith('module.') for k in state.keys()):
                state = {k[len('module.'):]: v for k, v in state.items()}
            
            gma_model.load_state_dict(state, strict=True)
            gma_model = gma_model.to(device)
            gma_model.eval()
            
            of_model = GMAWrapper(gma_model)
            of_model.requires_grad_(False)
            print("Using optical flow model: GMA")
        except Exception as e:
            print(f"Failed to load GMA: {e}")
            print("Falling back to RAFT model...")
            of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            of_model.requires_grad_(False)
            of_model = of_model.to(device)
            print("Using optical flow model: RAFT (fallback)")
    elif args.of_model == "DRFA":
        try:
            from opticalflow.gma_wrapper import GMAWrapper
            import sys
            
            if args.of_path is None:
                raise ValueError("--of_path is required for DRFA (path to GMA-SC source directory)")
            gma_path = args.of_path
            gma_core_path = os.path.join(gma_path, 'core')
            for p in [gma_path, gma_core_path]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            
            import core.update
            import gma
            from core.network import RAFTGMA
            
            class Args:
                def __init__(self):
                    self.mixed_precision = False
                    self.alternate_corr = False
                    self.dropout = 0.0
                    self.num_heads = 1
                    self.position_only = False
                    self.position_and_content = True
                    self.pe = 'linear'
                    self.self_pe = 'linear'
                    self.corr_levels = 4
                    self.corr_radius = 4
                def __contains__(self, key):
                    return hasattr(self, key)
            
            gma_args = Args()
            gma_model = RAFTGMA(gma_args)
            
            if args.of_pretrained is None:
                raise ValueError("--of_pretrained is required for DRFA (path to pretrained weights)")
            checkpoint = torch.load(args.of_pretrained, map_location='cpu')
            if 'model' in checkpoint:
                state = checkpoint['model']
            else:
                state = checkpoint
            
            if all(k.startswith('module.') for k in state.keys()):
                state = {k[len('module.'):]: v for k, v in state.items()}
            
            gma_model.load_state_dict(state, strict=True)
            gma_model = gma_model.to(device)
            gma_model.eval()
            
            of_model = GMAWrapper(gma_model)
            of_model.requires_grad_(False)
            print("Using optical flow model: DRFA")
        except Exception as e:
            print(f"Failed to load GMA: {e}")
            print("Falling back to RAFT model...")
            of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            of_model.requires_grad_(False)
            of_model = of_model.to(device)
            print("Using optical flow model: RAFT (fallback)")
    else:
        raise ValueError(f"Unsupported optical flow model: {args.of_model}. Please choose from 'RAFT', 'GMFlow', 'FlowFormerPP', 'GMA', or 'DRFA'")
    # =============================================================================================

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True)
    lpips_metric.requires_grad_(False)
    
    if not args.train_unet:
        unet.requires_grad_(False)
    if not args.train_controlnet:
        controlnet.requires_grad_(False)

    if args.train_controlnet:
        controlnet.train()
    else:
        controlnet.eval()
    
    disc.train()
    
    if args.train_unet:
        unet.train()
    else:
        unet.eval()

    # freeze discriminator backbone
    disc.model.requires_grad_(False)


    #if args.enable_xformers_memory_efficient_attention:
    #    if is_xformers_available():
    #        import xformers
#
    #        xformers_version = version.parse(xformers.__version__)
    #        if xformers_version == version.parse("0.0.16"):
    #            logger.warn(
    #                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #            )
    #        unet.enable_xformers_memory_efficient_attention()
    #        controlnet.enable_xformers_memory_efficient_attention()
    #    else:
    #        raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        disc.model.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation

    # params_to_optimize = controlnet.parameters()
    # optimizer = optimizer_class(
    #     params_to_optimize,
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon
    # )

    opt_class, opt_kwargs = build_opt(args.optimizer)
    
    g_params = []
    if args.train_controlnet:
        g_params.append({"params": controlnet.parameters(), "lr": args.G_lr})
    if args.train_unet:
        g_params.append({"params": unet.parameters(), "lr": args.G_lr})
    
    if not g_params:
        raise ValueError("At least one of --train_controlnet or --train_unet must be enabled")
    
    optimizer_G = opt_class(g_params, **opt_kwargs)

    optimizer_D = opt_class(
        disc.parameters(),
        lr=args.D_lr,
        **opt_kwargs
    )

    # train_dataset = make_train_dataset(args, tokenizer, accelerator)
    dataset_opts = OmegaConf.load(args.dataset_config_path)
    train_dataset = REDSDataset(dataset_opts['dataset']['train'])
    
    val_dataset = REDSDataset(dataset_opts['dataset']['test'])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_G,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, controlnet, optimizer_G, optimizer_D, train_dataloader, lr_scheduler, disc = accelerator.prepare(
        unet, controlnet, optimizer_G, optimizer_D, train_dataloader, lr_scheduler, disc
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    of_model.to(accelerator.device, dtype=weight_dtype)
    lpips_metric.to(accelerator.device)

    disc.to(accelerator.device, dtype=weight_dtype)
    # target_model.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    #================================================ Train! ===============================================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    
    best_val_score = float('-inf')
    best_checkpoint_path = None
    
    eval_metrics = None
    val_log_file = os.path.join(args.output_dir, "validation_scores.json")
    
    g_losses = []
    d_losses = []
    g_loss_steps = []
    d_loss_steps = []

    # Set initial phase to G. We switch between G and D
    phase = 'G'
    D_ts_list = []
    for cur_ts_item in args.D_ts.split(','):
        if '-' in cur_ts_item:
            start_ind, end_ind = cur_ts_item.split('-')
            D_ts_list += list(range(int(start_ind), int(end_ind)))
        else:
            D_ts_list.append(int(cur_ts_item))
    ts_D_choices = torch.tensor(D_ts_list, device=accelerator.device).long()

    timestep_list = np.linspace(1000, 0, num=args.num_ts, endpoint=False) - 1
    timestep_list = torch.tensor(timestep_list).long()
    timestep_list = timestep_list.to(accelerator.device)

    # get here input condition since it is fixed
    with torch.no_grad():
        tokenization = tokenizer([''] * args.train_batch_size, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        
        encoder_hidden_states = text_encoder(tokenization.input_ids.to(accelerator.device))[0]

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Prepare images
            # lq = batch['lq'] 
            gt = batch['gt']
            gt = 2 * gt - 1
            # lq = 2 * lq - 1
            lq, _ = apply_dynamic_degradation_batch(gt, gt.device, args.degradation_params_json)
            b, t, _, _, _ = lq.shape
            
            
            encoder_hidden_states_batch = encoder_hidden_states[:b].detach()
            
            #if args.esrgan:
            #    degraded_lq, degraded_sizes = apply_dynamic_degradation_batch(lq, lq.device, args.degradation_params_json)
            #    
            #    if global_step % 100 == 0:
            #        avg_size = sum(degraded_sizes) / len(degraded_sizes)
            #else:
            #    degraded_lq = lq
            #    degraded_sizes = [lq[i].nelement() * lq[i].element_size() for i in range(b)]
            #    if global_step % 100 == 0:
            #        avg_size = sum(degraded_sizes) / len(degraded_sizes)
            #
            #lq = degraded_lq
            # ----------------------------------------------------------------------------------------------
            
            upscaled_lq = rearrange(lq, 'b t c h w -> (b t) c h w')
            upscaled_lq = F.interpolate(upscaled_lq, scale_factor=4, mode='bicubic')
            upscaled_lq = rearrange(upscaled_lq, '(b t) c h w -> b t c h w', b=b, t=t)

            #encoder_hidden_states = encoder_hidden_states_base[:b]

            random_t = [round(random.random()) * 2 for _ in range(b)] # <- decide t-1 or t+1

            #gt_prev = torch.stack([gt[i, frame_idx] for i, frame_idx in enumerate(random_t)])
            #upscaled_lq_prev = torch.stack([upscaled_lq[i, frame_idx] for i, frame_idx in enumerate(random_t)])
            #lq_prev = torch.stack([lq[i, frame_idx] for i, frame_idx in enumerate(random_t)])

            gt_prev = torch.stack([gt[i, t] for i, t in enumerate(random_t)])
            upscaled_lq_prev = torch.stack([upscaled_lq[i, t] for i, t in enumerate(random_t)])
            lq_prev = torch.stack([lq[i, t] for i, t in enumerate(random_t)])

            gt = gt[:, t // 2, ...]
            lq = lq[:, t // 2, ...]
            upscaled_lq_cur = upscaled_lq[:, t // 2, ...]

            # Convert images to latent space
            latents = vae.encode(gt.to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents_prev = vae.encode(gt_prev.to(dtype=weight_dtype)).latent_dist.sample()
            latents_prev = latents_prev * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            ts_indices = torch.randint(0, args.num_ts, (bsz, ))
            timesteps = timestep_list[ts_indices].long()
            # print(timesteps)
            # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            # timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noisy_latents_cat = torch.cat([noisy_latents, lq], dim=1)

            # Get the text embedding for conditioning
            # tokenization = tokenizer('', max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            # encoder_hidden_states = text_encoder(tokenization)[0]


            # make prediction of the previous frame
            noise_prev = torch.randn_like(latents_prev)
            noisy_latents_prev = noise_scheduler.add_noise(latents_prev, noise_prev, timesteps)
            noisy_latents_prev_cat = torch.cat([noisy_latents_prev, lq_prev], dim=1)
            # noise_level = torch.cat([torch.tensor([20], dtype=torch.long, device=accelerator.device)] * b)
            if phase == 'G':
                models_to_accumulate = []
                if args.train_controlnet:
                    models_to_accumulate.append(controlnet)
                if args.train_unet:
                    models_to_accumulate.append(unet)
                
                with accelerator.accumulate(*models_to_accumulate):
                    disc.eval()
                    if args.train_controlnet:
                        controlnet.train()
                    if args.train_unet:
                        unet.train()
                    
                    model_pred_prev = unet(
                        noisy_latents_prev_cat,
                        timesteps,
                        # class_labels = noise_level,
                        encoder_hidden_states=encoder_hidden_states_batch
                    ).sample
                    approximated_x0_latent_prev = noise_scheduler.get_approximated_x0(model_pred_prev, timesteps, noisy_latents_prev)
                    approximated_x0_rgb_prev = vae.decode(approximated_x0_latent_prev / vae.config.scaling_factor).sample

                    # latents_prev_warped = compute_of_and_warp(of_model, upscaled_lq_cur, upscaled_lq_prev, latents_prev)
                    # controlnet_image = latents_prev_warped.to(dtype=weight_dtype)
                    f_flow = get_flow(of_model, upscaled_lq_cur, upscaled_lq_prev)
                    warped_approximated_x0 = flow_warp(approximated_x0_rgb_prev, f_flow)
                    controlnet_image = warped_approximated_x0.to(dtype=weight_dtype).detach()

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents_cat,
                        timesteps,
                        # class_labels = noise_level,
                        encoder_hidden_states=encoder_hidden_states_batch,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )
                    
                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents_cat,
                        timesteps,
                        # class_labels = noise_level,
                        encoder_hidden_states=encoder_hidden_states_batch,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample

                    approximated_x0_latent = noise_scheduler.get_approximated_x0(model_pred, timesteps, noisy_latents)
                    # add noise to generated latents and feed them to D
                    timesteps_D = ts_D_choices[torch.randint(0, len(ts_D_choices), (bsz, ), device=accelerator.device)]
                    noised_predicted_x0 = noise_scheduler.add_noise(approximated_x0_latent, torch.randn_like(latents), timesteps_D)

                    # adv loss
                    pred_fake = disc(noised_predicted_x0, timesteps_D, encoder_hidden_states_batch)
                    adv_loss = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
                    #recon loss
                    recon_loss = F.smooth_l1_loss(approximated_x0_latent, latents)
                    
                    # tLPIPS loss
                    approximated_x0_rgb = vae.decode(approximated_x0_latent / vae.config.scaling_factor).sample
                    gt_prev_01 = ((gt_prev + 1.0) / 2.0).clamp(0, 1)
                    gt_01 = ((gt + 1.0) / 2.0).clamp(0, 1)
                    approx_rgb_prev_01 = ((approximated_x0_rgb_prev + 1.0) / 2.0).clamp(0, 1)
                    approx_rgb_cur_01 = ((approximated_x0_rgb + 1.0) / 2.0).clamp(0, 1)
                    lpips_gt = lpips_metric(gt_01, gt_prev_01)
                    lpips_rec = lpips_metric(approx_rgb_cur_01, approx_rgb_prev_01)
                    tlpips_loss = (lpips_gt - lpips_rec).abs().mean()
                    
                    #total loss
                    loss = adv_loss * args.adv_weight + recon_loss * 1.0 + tlpips_loss * args.tlpips_weight


                    # Get the target for loss depending on the prediction type
                    # if noise_scheduler.config.prediction_type == "epsilon":
                    #     target = noise
                    # elif noise_scheduler.config.prediction_type == "v_prediction":
                    #     target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    # else:
                    #     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = []
                        if args.train_controlnet:
                            params_to_clip.extend(list(controlnet.parameters()))
                        if args.train_unet:
                            params_to_clip.extend(list(unet.parameters()))
                        grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()):
                            optimizer_G.zero_grad(set_to_none=True)
                            optimizer_D.zero_grad(set_to_none=True)
                            logger.warning("NaN/Inf detected, skipping iteration...")
                            continue

                        phase = 'D'

                    # optimizer.step()
                    optimizer_G.step()
                    lr_scheduler.step()
                    # optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    optimizer_G.zero_grad(set_to_none=True)
                    optimizer_D.zero_grad(set_to_none=True)
                logs = {'adv_loss': adv_loss.detach().item(), 
                        'recon_loss': recon_loss.detach().item(),
                        'tlpips_loss': tlpips_loss.detach().item(),
                        'lr': lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if accelerator.sync_gradients:
                    g_losses.append(loss.detach().item())
                    g_loss_steps.append(global_step)

            elif phase == 'D':
                with accelerator.accumulate(disc):
                    disc.train()
                    controlnet.eval()
                    unet.eval()
                    with torch.no_grad():
                        model_pred_prev = unet(
                        noisy_latents_prev_cat,
                        timesteps,
                        # class_labels = noise_level,
                        encoder_hidden_states=encoder_hidden_states_batch
                        ).sample
                        approximated_x0_latent_prev = noise_scheduler.get_approximated_x0(model_pred_prev, timesteps, noisy_latents_prev)
                        approximated_x0_rgb_prev = vae.decode(approximated_x0_latent_prev / vae.config.scaling_factor).sample

                        # latents_prev_warped = compute_of_and_warp(of_model, upscaled_lq_cur, upscaled_lq_prev, latents_prev)
                        # controlnet_image = latents_prev_warped.to(dtype=weight_dtype)
                        f_flow = get_flow(of_model, upscaled_lq_cur, upscaled_lq_prev)
                        warped_approximated_x0 = flow_warp(approximated_x0_rgb_prev, f_flow)
                        controlnet_image = warped_approximated_x0.to(dtype=weight_dtype).detach()

                        down_block_res_samples, mid_block_res_sample = controlnet(
                            noisy_latents_cat,
                            timesteps,
                            # class_labels = noise_level,
                            encoder_hidden_states=encoder_hidden_states_batch,
                            controlnet_cond=controlnet_image,
                            return_dict=False,
                        )
                    
                        # Predict the noise residual
                        model_pred = unet(
                            noisy_latents_cat,
                            timesteps,
                            # class_labels = noise_level,
                            encoder_hidden_states=encoder_hidden_states_batch,
                            down_block_additional_residuals=[
                                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                            ],
                            mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                        ).sample

                        approximated_x0_latent = noise_scheduler.get_approximated_x0(model_pred, timesteps, noisy_latents)
                    timesteps_D_fake = ts_D_choices[torch.randint(0, len(ts_D_choices), (bsz, ), device=accelerator.device)]
                    timesteps_D_real = ts_D_choices[torch.randint(0, len(ts_D_choices), (bsz, ), device=accelerator.device)]
                    noised_predicted_x0 = noise_scheduler.add_noise(approximated_x0_latent, torch.randn_like(latents), timesteps_D_fake)
                    noised_latents = noise_scheduler.add_noise(latents, torch.randn_like(latents), timesteps_D_real)

                    if args.misaligned_pairs_D and bsz > 1:
                        shifted_latents = torch.roll(latents, 1, 0)
                        timesteps_D_shifted_pairs = ts_D_choices[torch.randint(0, len(ts_D_choices), (bsz, ), device=accelerator.device)]
                        noised_shifted_latents = noise_scheduler.add_noise(shifted_latents, torch.randn_like(shifted_latents), timesteps_D_shifted_pairs)

                        noised_predicted_x0 = torch.concat([noised_predicted_x0, noised_shifted_latents], dim=0)
                        timesteps_D_fake = torch.concat([timesteps_D_fake, timesteps_D_shifted_pairs])

                    pred_fake = disc(noised_predicted_x0, timesteps_D_fake, encoder_hidden_states_batch)
                    pred_true = disc(noised_latents, timesteps_D_real, encoder_hidden_states_batch)


                    #calculate losses for fake and real data
                    loss_gen = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
                    loss_real = F.binary_cross_entropy_with_logits(pred_true, torch.ones_like(pred_true))
                    D_loss = loss_gen + loss_real

                    accelerator.backward(D_loss)
                    if accelerator.sync_gradients:
                        params_to_clip = disc.parameters()
                        grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()):
                            optimizer_G.zero_grad(set_to_none=True)
                            optimizer_D.zero_grad(set_to_none=True)
                            logger.warning("NaN/Inf detected, skipping iteration...")
                            continue
                        
                        phase = 'G'
            
                    optimizer_D.step()
                    optimizer_G.zero_grad(set_to_none=True)
                    optimizer_D.zero_grad(set_to_none=True)
                logs = {'D_loss': D_loss.detach().item(), 'loss_gen': loss_gen.detach().item(), 
                    'loss_real': loss_real.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if accelerator.sync_gradients:
                    d_losses.append(D_loss.detach().item())
                    d_loss_steps.append(global_step)        

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint") and not d.startswith("checkpoint-best")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        val_score, avg_metrics, eval_metrics = run_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            noise_scheduler,
                            val_dataloader,
                            accelerator,
                            weight_dtype,
                            of_model,
                            global_step,
                            eval_metrics=eval_metrics,
                            log_file=val_log_file
                        )
                        
                        if writer is not None:
                            writer.add_scalar('Validation/score', val_score, global_step)
                            writer.add_scalar('Validation/psnr', avg_metrics['psnr'], global_step)
                            writer.add_scalar('Validation/ssim', avg_metrics['ssim'], global_step)
                            writer.add_scalar('Validation/lpips', avg_metrics['lpips'], global_step)
                            writer.add_scalar('Validation/dists', avg_metrics['dists'], global_step)
                            writer.add_scalar('Validation/musiq', avg_metrics['musiq'], global_step)
                            writer.add_scalar('Validation/clip', avg_metrics['clip'], global_step)
                            writer.add_scalar('Validation/tlpips', avg_metrics['tlpips'], global_step)
                            writer.add_scalar('Validation/tof', avg_metrics['tof'], global_step)
                        
                        accelerator.log({
                            "val_score": val_score,
                            "val_psnr": avg_metrics['psnr'],
                            "val_ssim": avg_metrics['ssim'],
                            "val_lpips": avg_metrics['lpips']
                        }, step=global_step)
                        
                        if val_score > best_val_score:
                            best_val_score = val_score
                            
                            if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                                shutil.rmtree(best_checkpoint_path)
                            
                            best_checkpoint_path = os.path.join(args.output_dir, f"checkpoint-best")
                            accelerator.save_state(best_checkpoint_path)
                            logger.info(f"New best model saved with val_score={best_val_score:.4f} at {best_checkpoint_path}")
                            
                            with open(os.path.join(best_checkpoint_path, "best_info.txt"), "w") as f:
                                f.write(f"Step: {global_step}\n")
                                f.write(f"Validation Score: {best_val_score:.4f}\n")
                                f.write(f"PSNR: {avg_metrics['psnr']:.3f}\n")
                                f.write(f"SSIM: {avg_metrics['ssim']:.4f}\n")
                                f.write(f"LPIPS: {avg_metrics['lpips']:.4f}\n")
                                f.write(f"DISTS: {avg_metrics['dists']:.4f}\n")

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if g_losses or d_losses:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            if g_losses:
                plt.plot(g_loss_steps, g_losses, label='G Loss', alpha=0.7, linewidth=1)
            
            if d_losses:
                plt.plot(d_loss_steps, d_losses, label='D Loss', alpha=0.7, linewidth=1)
            
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            loss_curve_path = os.path.join(args.output_dir, "loss_curves.png")
            plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Loss curves saved to {loss_curve_path}")
        
        if args.train_controlnet:
            controlnet = accelerator.unwrap_model(controlnet)
            controlnet.save_pretrained(os.path.join(args.output_dir, "controlnet"))
        
        if args.train_unet:
            unet = accelerator.unwrap_model(unet)
            unet.save_pretrained(os.path.join(args.output_dir, "unet"))

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

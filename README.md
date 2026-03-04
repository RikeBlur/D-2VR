# D²-VR: Degradation-Robust and Distilled Video Restoration with Synergistic Optimization Strategy

This repository contains the official PyTorch implementation of D²-VR.

<img width="1723" height="978" alt="image" src="https://github.com/user-attachments/assets/3d640a10-4dcd-4869-b7b6-f9084636290d" />

# Some Information

[Paper](https://arxiv.org/pdf/2602.08395) · [Project Page (coming soon)](www.example.com)

# News

* [2026.2] Our pre-print paper is released on arXiv.

## Architecture Overview

- **UNet + ControlNet**: The UNet predicts the denoised current frame; the ControlNet receives the warped reconstruction of the adjacent frame as conditional input.
- **Optical Flow**: Supports RAFT, GMFlow, FlowFormer++, GMA, and DRFA (default).
- **Discriminator**: A UNet-based discriminator is used for adversarial training.
- **Dynamic Degradation**: During training, GT frames are degraded on-the-fly using configurable blur / noise / JPEG parameters (see `degradation/degradation_params.json`).

## Environment

Python 3.8+, CUDA 11, and [diffusers](https://github.com/huggingface/diffusers).

```bash
conda create -n d2vr python=3.8 -y
conda activate d2vr
pip install -r requirements.txt
```

## Datasets

Download the [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset (sharp + bicubic low-resolution split).

Data layout expected:
```
REDS/
  train/<seq_id>/<frame>.png   # GT
REDS-BI/
  train/<seq_id>/<frame>.png   # LR (bicubic)
```

Generate the metadata files required by the dataloader:
```bash
python create_metadata.py --gt_root /path/to/REDS --lq_root /path/to/REDS-BI --output_dir ./dataset/REDS
```

Then update the paths in `dataset/REDS/config_test.yaml` accordingly.

## Training

Adjust paths and hyperparameters in `train.sh`, then run:

```bash
bash ./train.sh
```

Example command:

```bash
python train.py \
  --pretrained_model_name_or_path="claudiom4sir/StableVSR" \
  --pretrained_vae_model_name_or_path="claudiom4sir/StableVSR" \
  --controlnet_model_name_or_path="/path/to/controlnet/" \
  --unet_model_name_or_path="/path/to/unet/" \
  --output_dir="/path/to/output/" \
  --logging_dir="/path/to/output/logs/" \
  --dataset_config_path="./dataset/REDS/config_test.yaml" \
  --degradation_params_json ./degradation/degradation_params.json \
  --G_lr=1e-5 \
  --D_lr=1e-5 \
  --validation_steps=1000 \
  --train_batch_size=4 \
  --adv_weight=0.05 \
  --tlpips_weight=0.1 \
  --num_ts=4 \
  --dataloader_num_workers=8 \
  --max_train_steps=40000 \
  --checkpointing_steps=1000 \
  --train_unet \
  --train_controlnet \
  --of_model="DRFA"
```

Key training arguments:

| Argument | Description |
|---|---|
| `--train_unet` | Enable UNet fine-tuning |
| `--train_controlnet` | Enable ControlNet training |
| `--of_model` | Optical flow model: `RAFT`, `GMFlow`, `FlowFormerPP`, `GMA`, `DRFA` |
| `--adv_weight` | Weight for adversarial loss |
| `--tlpips_weight` | Weight for temporal LPIPS loss |
| `--num_ts` | Number of timesteps sampled per iteration |

### Memory Requirements

Training with the default configuration (batch size 4, resolution 256) requires approximately **17 GB** GPU memory.

## Inference

```bash
python test.py \
  --in_path /path/to/lr_sequences/ \
  --num_inference_steps 4 \
  --controlnet_ckpt /path/to/controlnet_checkpoint/ \
  --unet_ckpt /path/to/unet/ \
  --model_path /path/to/base_model/
```

Example:
```bash
python test.py --in_path /remote-home/share/liangjianfeng/REDS-BI/val-10/ --num_inference_steps 4 --controlnet_ckpt /remote-home/share/liangjianfeng/stablevsr/ckpt/REDS_SC/checkpoint-10000/controlnet/ --unet_ckpt /remote-home/share/liangjianfeng/stablevsr/StableVSR/unet/ --model_path /remote-home/share/liangjianfeng/stablevsr/StableVSR/
```

Key inference arguments:

| Argument | Description |
|---|---|
| `--in_path` | Path to folder containing LR video sequences |
| `--out_path` | Output folder (default: `./results/`) |
| `--num_inference_steps` | Number of denoising steps |
| `--controlnet_ckpt` | Path to the ControlNet checkpoint folder |
| `--unet_ckpt` | Path to the UNet checkpoint folder |
| `--model_path` | Path to the base diffusion model (VAE, text encoder, tokenizer) |
| `--of_model` | Optical flow model to use (default: `DRFA`) |
| `--of_pretrained` | Pretrained Optical flow model to use |
| `--scheduler` | Noise scheduler: `DDPM` (default), `EulerAncestralDiscrete`, `LMSDiscrete` |

## Evaluation

Full-reference evaluation (PSNR / SSIM / LPIPS / DISTS / tLPIPS / tOF / MUSIQ / MANIQA / CLIP-IQA / NIQE):

```bash
python eval.py --gt_path /path/to/gt_sequences/ --out_path /path/to/results/
```

No-reference evaluation only:

```bash
python eval.py --out_path /path/to/results/ --no_reference
```

Save metrics to JSON:

```bash
python eval.py --gt_path /path/to/gt/ --out_path /path/to/results/ --json_output metrics.json
```
#!/usr/bin/env bash
python train.py --name gma-degradation --stage sintel --validation sintel --output results/sintel/gma --restore_ckpt checkpoints/gma-things.pth --num_steps 20000 --lr 1e-5 --image_size 368 768 --wdecay 0.0001 --gamma 0.85 --gpus 0 1 --batch_size 16 --val_freq 1000 --print_freq 100 --mixed_precision

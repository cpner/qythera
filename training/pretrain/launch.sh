#!/bin/bash
set -e
NUM_GPUS=${1:-4}
CONFIG=${2:-training/configs/7b_lora.yaml}
echo "Starting pretraining with $NUM_GPUS GPUs"
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 training/pretrain/train_pretrain.py --config "$CONFIG" --deepspeed training/pretrain/deepspeed_config.json

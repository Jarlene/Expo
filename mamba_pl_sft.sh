#! /usr/bin/env bash
set -e

#tokenizer_path=/home/work/xiongwenlong/models/chatglm3-6b-32k
dataset_dir=/home/work/xiongwenlong/data/wudao
model_dir=/home/work/xiongwenlong/models

devices=0,1,2,3
arr=(`echo $devices | tr ',' ' '`)
num_proc=${#arr[@]}

model_name=$1
pretrained_model="${model_dir}/${model_name}"

OMP_NUM_THREADS=32  torchrun --standalone --nproc_per_node=$num_proc  mamba_tune.py \
    --model_name_or_path $pretrained_model \
    --data_dir $dataset_dir \
    --batch_size 2 \
    --lr 0.0001 \
    --output_dir output/pl \
    --log_dir logs/pl \
    --name $model_name \
    --num_proc $num_proc \
    --strategy deepspeed_stage_2_offload \
    --devices $devices \
    --quantizer false \
    --num_epochs 1 \
    --version v2 \
    --log_every_n_steps 10 \
    --val_check_interval 100 \
    --num_proc $num_proc \
    --d_state 8 \
    --beta 0.6


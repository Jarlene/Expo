#! /usr/bin/env bash
set -e

#tokenizer_path=/home/work/xiongwenlong/models/chatglm3-6b-32k
model_dir=/home/work/xiongwenlong/models
dataset_dir=gbharti/finance-alpaca

devices=0,1,2,3
arr=(`echo $devices | tr ',' ' '`)
num_proc=${#arr[@]}

model_name=$1
pretrained_model="${model_dir}/${model_name}"

OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=$devices torchrun --standalone --nproc_per_node=$num_proc  moe_tune.py \
    --model_name_or_path $pretrained_model \
    --data_dir $dataset_dir \
    --batch_size 2 \
    --lr 0.0001 \
    --output_dir output/pl \
    --log_dir logs/pl \
    --name ${model_name}_moe \
    --num_proc $num_proc \
    --strategy deepspeed_stage_2_offload \
    --quantizer false \
    --precision bf16 \
    --num_epochs 10 \
    --version v1 \
    --log_every_n_steps 10 \
    --val_check_interval 100 \
    --num_proc $num_proc \
    --num_experts 4 \
    --num_experts_per_token 2 \
    --router_jitter_noise 0.2 \
    --beta 0.6 \
    --num_slots 32 \
    --use_soft_moe true 


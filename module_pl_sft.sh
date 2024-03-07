#! /usr/bin/env bash
set -e

#tokenizer_path=/home/work/xiongwenlong/models/chatglm3-6b-32k
model_dir=/home/work/xiongwenlong/models
dataset_dir=gbharti/finance-alpaca

devices=0,1,2,3
arr=(`echo $devices | tr ',' ' '`)
num_proc=${#arr[@]}

model_name="Mistral-7B-Instruct"
pretrained_model="${model_dir}/${model_name}"

OMP_NUM_THREADS=16  CUDA_VISIBLE_DEVICES=$devices  torchrun --standalone --nproc_per_node=$num_proc  module_tune.py \
    --model_name_or_path $pretrained_model \
    --data_dir $dataset_dir \
    --batch_size 4 \
    --lr 0.0001 \
    --output_dir output/pl \
    --log_dir logs/pl \
    --name $model_name \
    --num_proc $num_proc \
    --strategy deepspeed_stage_2 \
    --quantizer false \
    --precision bf16 \
    --num_epochs 10 \
    --version v8 \
    --log_every_n_steps 1 \
    --val_check_interval 100 \
    --num_proc $num_proc \
    --d_state 8 \
    --beta 0.6 \
    --max_length 1024 \
    --val_data_percentage 0.01


#! /usr/bin/env bash
set -e

tokenizer_path=/home/work/xiongwenlong/models/chatglm3-6b-32k
dataset_dir=gbharti/finance-alpaca

devices=0,1,2,3
arr=(`echo $devices | tr ',' ' '`)
num_proc=${#arr[@]}

OMP_NUM_THREADS=8  torchrun --standalone --nproc_per_node=$num_proc model_pl_pt.py \
    --tokenizer_path $tokenizer_path \
    --data_dir $dataset_dir \
    --batch_size 10 \
    --output_dir output/pl \
    --name blink \
    --precision bf16-mixed \
    --log_every_n_steps 10 \
    --val_check_interval 100 \
    --strategy deepspeed_stage_2 \
    --devices $devices \
    --num_epochs 1 \
    --version v1 \
    --d_state 8 \
    --num_hidden_layers  6 \


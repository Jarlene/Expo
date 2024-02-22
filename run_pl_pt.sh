#! /usr/bin/env bash
set -e

tokenizer_path=/home/work/xiongwenlong/models/chatglm3-6b-32k
dataset_dir=/home/work/xiongwenlong/data/wudao

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 model_pl_pt.py \
    --tokenizer_path $tokenizer_path \
    --data_dir $dataset_dir \
    --batch_size 10 \
    --output_dir output/pl \
    --name blink \
    --precision bf16-mixed \
    --log_every_n_steps 10 \
    --val_check_interval 100 \
    --strategy deepspeed_stage_2 \
    --num_epochs 1 \
    --version v1 \
    --d_state 8 \
    --num_hidden_layers  6 \


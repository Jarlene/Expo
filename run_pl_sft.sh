#! /usr/bin/env bash
set -e

#tokenizer_path=/home/work/xiongwenlong/models/chatglm3-6b-32k
dataset_dir=gbharti/finance-alpaca
model_dir=/home/work/xiongwenlong/models

lora_trainable="gate_proj,up_proj,down_proj"
modules_to_save="lm_head,embed_tokens"
lora_rank=16
lora_alpha=32
lora_dropout=0.05

devices=0,1,2,3
arr=(`echo $devices | tr ',' ' '`)
num_proc=${#arr[@]}

model_name=$1
pretrained_model="${model_dir}/${model_name}"

OMP_NUM_THREADS=8  torchrun --standalone --nproc_per_node=$num_proc model_pl_sft.py \
    --model_name_or_path $pretrained_model \
    --data_dir $dataset_dir \
    --batch_size 4 \
    --output_dir output/pl \
    --name $model_name \
    --strategy deepspeed_stage_2 \
    --devices $devices \
    --quantizer true \
    --num_epochs 1 \
    --version v1 \
    --precision bf16-mixed \
    --num_experts 4 \
    --log_every_n_steps 10 \
    --val_check_interval 100 \
    --lora_rank $lora_rank \
    --num_proc $num_proc \
    --lora_alpha $lora_alpha \
    --lora_trainable $lora_trainable \
    --modules_to_save $modules_to_save \
    --num_experts_per_token 2 \
    --num_experts 4 
  


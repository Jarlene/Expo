#! /usr/bin/env bash
set -e

lr=1e-4
lora_rank=16
lora_alpha=32
#lora_trainable=".*.experts.\d+.(gate_proj|down_proj|up_proj)$"
lora_trainable="gate_proj,down_proj,up_proj"
lora_dropout=0.05
modules_to_save="lm_head,embed_tokens"

model_dir=/home/work/xiongwenlong/models
tokenizer_path=/home/work/xiongwenlong/models/chatglm3-6b-32k
dataset_dir=gbharti/finance-alpaca

per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=4

deepspeed_config_file=config/ds_zero2_no_offload.json

devices=0,1,2,3
arr=(`echo $devices | tr ',' ' '`)
num_proc=${#arr[@]}

model_name=$1
pretrained_model="${model_dir}/${model_name}"
version="v1"

accelerate launch --num_machines=1  --gpu_ids=$devices --mixed_precision=fp16 --main_process_port=9512  --num_processes=$num_proc model_hf_sft.py \
    --model_name_or_path $pretrained_model \
    --data_dir $dataset_dir \
    --output_dir "output/hf/$model_name/$version" \
    --deepspeed $deepspeed_config_file \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --do_train \
    --learning_rate 0.00001 \
    --do_eval \
    --run_name $model_name \
    --data_seed 11224 \
    --seed $RANDOM \
    --fp16 \
    --val_data_percentage 0.0001 \
    --num_train_epochs 1 \
    --logging_dir "logs/hf/$model_name/$version" \
    --lr_scheduler_type cosine \
    --weight_decay 0  \
    --warmup_steps 2 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 200 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_num_workers $num_proc \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step true \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --lora_trainable $lora_trainable \
    --modules_to_save $modules_to_save \
    --lora_dropout $lora_dropout \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --remove_unused_columns false \
    --quantizer true \
    --num_experts 4 \
    --num_experts_per_token 2 \
    --max_length 4096
#    --resume_from_checkpoint /home/work/xiongwenlong/models/Mistral-7B-Instruct/output/checkpoint-200

#! /usr/bin/env bash
set -e

lr=1e-4
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
version="v1"

accelerate launch --num_machines=1  --gpu_ids=$devices --mixed_precision=fp16 --main_process_port=6512  --num_processes=$num_proc model_hf_pt.py \
    --model_name $model_name \
    --data_dir $dataset_dir \
    --tokenizer_path $tokenizer_path \
    --output_dir "output/hf/$model_name/$version" \
    --deepspeed $deepspeed_config_file \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --do_train \
    --learning_rate $lr \
    --do_eval \
    --fp16 \
    --run_name $model_name \
    --data_seed 11224 \
    --seed $RANDOM \
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
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step true \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --remove_unused_columns false \
    --max_length 4096 \
    --intermediate_size 4096 \
    --hidden_size 1024 \
    --num_hidden_layers 6 \
    --d_conv 8
#    --resume_from_checkpoint /home/work/xiongwenlong/models/Mistral-7B-Instruct/output/checkpoint-200


# -*- coding: UTF-8 -*-
import os
from flask import Flask, request, jsonify
import torch
from typing import List, Dict


from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

generator: AutoModelForCausalLM = None
tokenizer: AutoTokenizer = None

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


def load_model(peft_path: str):
    config = PeftConfig.from_pretrained(peft_path)
    model_id = config.base_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    peft_model = PeftModel.from_pretrained(
        model,
        peft_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto',
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,  trust_remote_code=True,)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return peft_model, tokenizer


@app.before_request
def prepare():
    global generator, tokenizer
    if not generator:
        peft_path = '/home/work/xiongwenlong/models/Expo/output/hf/Mistral-7B-Instruct/v1/checkpoint-200'
        generator, tokenizer = load_model(peft_path)


def generate(messages: List[List[Dict]], temperature=0.5, max_new_tokens=4096, top_p=0.7, repetition_penalty=1.0):
    global generator, tokenizer
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
    )
    prompt_ids = []
    if not isinstance(messages[0], list):
        messages = [messages]
    for m in messages:
        prompt = tokenizer.apply_chat_template(m)
        prompt_ids.append(prompt)

    batch_max_len = max([len(ele) for ele in prompt_ids])
    attention_mask_batch = []
    input_ids_batch_padding = []
    for input_ids in prompt_ids:
        len_diff = batch_max_len - len(input_ids)
        input_ids = [tokenizer.pad_token_id]*batch_max_len + input_ids
        input_ids = input_ids[-batch_max_len:]  # truncate left
        mask = [0]*batch_max_len
        mask[len_diff:] = [1]*(batch_max_len-len_diff)
        attention_mask_batch.append(mask)
        input_ids_batch_padding.append(input_ids)
    input_ids = torch.LongTensor(input_ids_batch_padding).cuda()
    attention_mask = torch.BoolTensor(attention_mask_batch).cuda()
    res = generator.generate(input_ids, attention_mask=attention_mask,
                             **generate_kwargs)
    response_batch = []
    for r in res:
        result = tokenizer.decode(r[batch_max_len:], skip_special_tokens=True)
        response_batch.append(result)
    return response_batch


@app.route('/generate', methods=['POST'])
def gen():
    data = request.get_json()
    messages = data['messages']
    temperature = data.get('temperature', 0.3)
    top_p = data.get('top_p', 0.85)
    max_new_tokens = data.get('max_new_tokens', 4096)

    output = generate(messages, temperature=temperature,
                      top_p=top_p, max_new_tokens=max_new_tokens)

    res = {'choices': [{'message': {'content': output}}], 'status': 200}
    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True, host="10.38.3.53", port=8888)

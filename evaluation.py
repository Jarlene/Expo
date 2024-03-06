import lm_eval
from lm_eval.models.huggingface import HFLM
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_peft_model(peft_path: str, device: str):
    config = PeftConfig.from_pretrained(peft_path)
    model_id = config.base_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    peft_model = PeftModel.from_pretrained(
        model,
        peft_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        adapter_name='sft',
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,  trust_remote_code=True,)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(peft_model)
    lm = HFLM(pretrained=peft_model, tokenizer=tokenizer,
              batch_size=16, trust_remote_code=True)
    return lm


def get_model(model_path: str, device: str):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,  trust_remote_code=True,)
    print(model)
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
              batch_size=16, trust_remote_code=True)
    return lm


if __name__ == "__main__":
    eval_models = {
        'moe_lora': '/home/work/xiongwenlong/models/Expo/output/hf/Mistral-7B-Instruct/v7/checkpoint-200/sft',
        'lora': '/home/work/xiongwenlong/models/Expo/output/hf/Mistral-7B-Instruct/v5/checkpoint-100/sft',
        'mov': '/home/work/xiongwenlong/models/Expo/output/hf/Mistral-7B-Instruct/v8/checkpoint-5300/sft'
    }
    device = 'cuda:0'
    for model_type, peft_path in eval_models.items():
        lm = get_peft_model(peft_path, device=device)
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=["mmlu", "gsm8k", "triviaqa"],
            num_fewshot=0,
            device=device,
            batch_size=16,
            write_out=True,
            task_manager=lm_eval.tasks.TaskManager(),)
        print(results)

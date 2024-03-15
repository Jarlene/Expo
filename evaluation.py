from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer
import json
import numpy as np
from utils.utils import TrainArguments, get_train_args
import os


@dataclass
class ScriptArguments(TrainArguments):

    model_name_or_path: str = field(
        metadata={"help": "the location of the SFT model name or path"},

    )
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "tokenizer path"}
    )

    save_dir: str = field(
        default=None,
        metadata={"help": "result save dir"}
    )

    quantizer: Optional[bool] = field(
        default=False,
        metadata={"help": "load model quantizer"}
    )

    tasks: str = field(
        metadata={'help', 'evalution task names, split with `,`'}
    )

    cache_requests: Optional[bool] = field(
        default=False,
        metadata={"help": "cache requests"}
    )

    temperature: Optional[float] = field(
        default=0.0,
        metadata={"help": "temperature"}
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "do sample"}
    )
    top_p: Optional[float] = field(
        default=0.3,
        metadata={"help": "top p"}
    )
    repetition_penalty: Optional[float] = field(
        default=1.1,
        metadata={"help": "repetition penalty"}
    )
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={"help": "max new tokens"}
    )

    use_gen_kwargs: Optional[bool] = field(
        default=False,
        metadata={"help": "use gen kwargs"}
    )

    adapter_name: Optional[str] = field(
        default='default',
        metadata={"help": "adapter name"}
    )

    num_fewshot: Optional[int] = field(
        default=0,
        metadata={"help": "num fewshot"}
    )


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def get_peft_model(script_args: ScriptArguments):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    config = PeftConfig.from_pretrained(script_args.model_name_or_path)
    model_id = config.base_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if script_args.quantizer else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    peft_model = PeftModel.from_pretrained(
        model,
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        adapter_name=script_args.adapter_name,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,  trust_remote_code=True,)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(peft_model)
    lm = HFLM(pretrained=peft_model,
              tokenizer=tokenizer,
              dtype=torch.bfloat16,
              max_length=tokenizer.model_max_length,
              batch_size=script_args.batch_size,
              trust_remote_code=True)
    return lm


def build_gen_kwargs(script_args: ScriptArguments):
    if script_args.use_gen_kwargs:
        res = f"temperature={script_args.temperature},do_sample={script_args.do_sample},top_p={script_args.top_p},repetition_penalty={script_args.repetition_penalty},max_new_tokens={script_args.max_new_tokens}"
        return res
    return None


def get_model(script_args: ScriptArguments):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        quantization_config=bnb_config if script_args.quantizer else None,
        torch_dtype=torch.bfloat16,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
    )
    print(model)
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
              dtype=torch.bfloat16,
              max_length=tokenizer.model_max_length,
              batch_size=script_args.batch_size,
              trust_remote_code=True,
              )
    return lm


def eval():
    script_args = get_train_args(ScriptArguments)
    os.environ["CUDA_VISIBLE_DEVICES"] = script_args.devices
    if script_args.save_dir is None:
        script_args.save_dir = f"./"
    tasks = script_args.tasks.split(',')
    try:
        lm = get_peft_model(script_args.model_name_or_path,
                            device=script_args.devices, batch_size=script_args.batch_size)
    except Exception as e:
        lm = get_model(script_args.model_name_or_path,
                       device=script_args.devices, batch_size=script_args.batch_size)

    if lm is None:
        raise ValueError("lm is None")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=script_args.num_fewshot,
        device=script_args.devices,
        cache_requests=script_args.cache_requests,
        write_out=True,
        gen_kwargs=build_gen_kwargs(script_args),
        task_manager=lm_eval.tasks.TaskManager(),
    )

    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))

    dumped = json.dumps(
        results, indent=2, default=_handle_non_serializable, ensure_ascii=False
    )
    if not os.path.exists(f"{script_args.save_dir}/{script_args.name}/{script_args.version}"):
        os.makedirs(
            f"{script_args.save_dir}/{script_args.name}/{script_args.version}", exist_ok=True)
    with open(f"{script_args.save_dir}/{script_args.name}/{script_args.version}/results.json", 'w', encoding='utf-8') as f:
        f.write(dumped)


if __name__ == "__main__":
    eval()

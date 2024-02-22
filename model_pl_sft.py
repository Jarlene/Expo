import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
from trainer.trainer import get_trainer
from utils.utils import TrainArguments, get_train_args
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import load_from_disk, load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, BitsAndBytesConfig

from peft import MoELoraConfig, get_peft_model,prepare_model_for_kbit_training
torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_prompt(data_point):
    """
    Generate input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenized prompt
    """

    if data_point['input']:
        text = 'Below is an instruction that describes a task, paired with an input that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Input:\n{data_point["input"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'

    else:
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'
    return text


@dataclass
class ScriptArguments(TrainArguments):
    """
    The arguments for the sft training script.
    """

    quantizer: Optional[bool] = field(default=None,
                                      metadata={
                                          "help": "load model quantizer"})
    num_experts: Optional[int] = field(
        default=0, metadata={"help": "the num of experts"})
    router_jitter_noise: Optional[float] = field(
        default=0.2, metadata={"help": "router_jitter_noise"})
    num_experts_per_token: Optional[int] = field(
        default=3, metadata={"help": "num_experts_per_token"})

    max_length: Optional[int] = field(
        default=8192, metadata={"help": "max length for model input"})

    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_rank: Optional[int] = field(
        default=8, metadata={"help": "the lora r parameter"})
    lora_trainable: Optional[str] = field(default='q_proj,k_proj,v_proj', metadata={
                                          'help': "lora trainable params"})
    modules_to_save: Optional[str] = field(default=None, metadata={
                                           'help': "lora  params to save"})

    adapter_name: Optional[str] = field(
        default='default', metadata={'help': "adapter name"})


@dataclass
class HugeDataCollator(DefaultDataCollator):

    tokenizer: AutoTokenizer = None
    max_length: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        contents = [generate_prompt(d) for d in features]
        output = self.tokenizer(contents,
                                padding='longest',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='pt')
        output['labels'] = output['input_ids'].clone()
        return output


def get_tokenizer(script_args: ScriptArguments):
    need_resize_embed = False
    if script_args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.tokenizer_path, padding_side="left", trust_remote_code=True)
        need_resize_embed = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name_or_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, need_resize_embed


def get_lora_config(script_args: ScriptArguments):
    targets = script_args.lora_trainable.split(',')
    if len(targets) == 1:
        targets = targets[0]
    modules = None
    if script_args.modules_to_save:
        modules = script_args.modules_to_save.split(',')
        if len(modules) == 1:
            modules = modules[0]
    peft_config = MoELoraConfig(
        r=script_args.lora_rank,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=targets,
        modules_to_save=modules,
        task_type="CAUSAL_LM",
        bias='none',
        router_jitter_noise=script_args.router_jitter_noise,
        num_experts=script_args.num_experts,
        num_experts_per_token=script_args.num_experts_per_token,
    )

    return peft_config


def get_model_and_tokenizer(script_args: ScriptArguments, trainable=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=bnb_config if script_args.quantizer else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if trainable:
        model.config.use_cache = False
    peft_config = get_lora_config(script_args)

    model = get_peft_model(
        model, peft_config, adapter_name=script_args.adapter_name)
    model.print_trainable_parameters()
    print(model)
    model.config.num_experts = script_args.num_experts
    model.config.router_jitter_noise = script_args.router_jitter_noise
    model.config.num_experts_per_token = script_args.num_experts_per_token
    tokenizer, need_resize_embed = get_tokenizer(script_args)
    if need_resize_embed:
        model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tokenizer


def get_data(script_args: ScriptArguments):
    if script_args.data_dir.startswith('/'):
        raw_data = load_from_disk(script_args.data_dir)
    else:
        raw_data = load_dataset(script_args.data_dir, split='train')
    if isinstance(raw_data, DatasetDict):
        raw_data = raw_data['train']
    data = raw_data.train_test_split(
        test_size=script_args.val_data_percentage)
    return data['train'], data['test']


def main():
    script_args = get_train_args(ScriptArguments)
    train_dataset, eval_dataset = get_data(script_args)
    model, tokenizer = get_model_and_tokenizer(script_args, trainable=True)
    data_collator = HugeDataCollator(
        tokenizer=tokenizer, max_length=script_args.max_length)
    trainer = get_trainer(
        args=script_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        tokenizer=tokenizer,
        collate_fn=data_collator)
    trainer.train()
    trainer.save_pretrained()


if __name__ == "__main__":
    main()

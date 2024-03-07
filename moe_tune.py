from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
import torch
import os
from datasets import load_from_disk, load_dataset, DatasetDict

from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, BitsAndBytesConfig
from sklearn.metrics import accuracy_score
from utils.utils import get_train_args, TrainArguments
from trainer.trainer import get_trainer
from layer.moe.moe import DroplessMoE, SoftMoE
from peft import prepare_model_for_kbit_training
import copy

torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def accuracy(predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(
            accuracy_score(references, predictions,
                           normalize=normalize, sample_weight=sample_weight)
        )
    }


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


@dataclass
class HugeDataCollator(DefaultDataCollator):

    tokenizer: AutoTokenizer = None
    max_length: int = None
    generate_func: Any = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        contents = [self.generate_func(d) for d in features]
        output = self.tokenizer(contents,
                                padding='longest',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='pt')
        output['labels'] = output['input_ids'].clone()
        return output


@dataclass
class ScriptArguments(TrainArguments):
    """
    The arguments for the sft training script.
    """

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "data dir "},
    )

    val_data_percentage: Optional[float] = field(
        default=0.00001,
        metadata={"help": "train test split ratio"})

    tokenizer_path: str = field(
        default=None,
        metadata={"help": "tokenizer path"})

    max_length: Optional[int] = field(
        default=8192, metadata={"help": "max length for model input"})

    use_soft_moe: Optional[bool] = field(
        default=True, metadata={"help": "use soft moe"}
    )
    num_experts: Optional[int] = field(
        default=3, metadata={"help": "the num of experts"})
    router_jitter_noise: Optional[float] = field(
        default=0.2, metadata={"help": "router_jitter_noise"})

    num_experts_per_token: Optional[int] = field(
        default=1, metadata={"help": "num_experts_per_token"})

    num_slots: Optional[int] = field(
        default=16, metadata={"help": "soft moe for slots num"})

    beta: Optional[float] = field(
        default=1.0,
        metadata={"help": "ssm model expand"}
    )

    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_rank: Optional[int] = field(
        default=8, metadata={"help": "the lora r parameter"})


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


class LoRABlock(torch.nn.Module):
    def __init__(self, hidden_size, rank, lora_alpha, lora_dropout) -> None:
        super().__init__()
        self.loar_A = torch.nn.Linear(hidden_size, rank, bias=False)
        self.loar_B = torch.nn.Linear(rank, hidden_size, bias=False)
        self.scaling = lora_alpha / rank
        self.lora_dropout = torch.nn.Dropout(p=lora_dropout)

    def forward(self, hidden_states):
        x = self.lora_dropout(hidden_states)
        x = self.loar_A(x)
        x = self.loar_B(x)
        return x * self.scaling


class MoEHook(torch.nn.Module):

    def __init__(self, base_layer: torch.nn.Module, hidden_size,
                 script_args: ScriptArguments) -> None:
        super().__init__()
        base_layer.requires_grad_(False)
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        if script_args.use_soft_moe:
            self.moe = SoftMoE(
                hidden_size=hidden_size,
                num_experts=script_args.num_experts,
                num_slots=script_args.num_slots,
                router_jitter_noise=script_args.router_jitter_noise,
                expert=base_layer)
        else:
            self.moe = DroplessMoE(hidden_size=hidden_size, num_experts=script_args.num_experts,
                                   num_experts_per_token=script_args.num_experts_per_token,
                                   router_jitter_noise=script_args.router_jitter_noise,
                                   expert=base_layer)
        self.beta = script_args.beta
        self.use_soft_moe = script_args.use_soft_moe

    def forward(self, x: torch.Tensor):
        result = self.base_layer(x)
        if self.use_soft_moe:
            domain = self.moe(x)
        else:
            domain, _, _ = self.moe(x)
        return result + self.beta * domain


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
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)
    if trainable:
        model.config.use_cache = False
    tokenizer, need_resize_embed = get_tokenizer(script_args)
    if need_resize_embed:
        model.resize_token_embeddings(len(tokenizer))

    for idx, child in enumerate(model.model.layers):
        base_layer = child.mlp
        if idx % 2 == 0:
            child.mlp = MoEHook(
                base_layer, model.config.hidden_size, script_args)
        else:
            child.mlp = base_layer
    print(model)
    model.config.num_experts = script_args.num_experts
    model.config.num_experts_per_token = script_args.num_experts_per_token
    model.config.router_jitter_noise = script_args.router_jitter_noise
    model.config.beta = script_args.beta
    model.config.use_soft_moe = script_args.use_soft_moe
    model.config.num_slots = script_args.num_slots
    return model, tokenizer


def main():
    script_args = get_train_args(ScriptArguments)
    train_dataset, eval_dataset = get_data(script_args)
    model, tokenizer = get_model_and_tokenizer(script_args, trainable=True)
    data_collator = HugeDataCollator(
        tokenizer=tokenizer, max_length=script_args.max_length, generate_func=generate_prompt)
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

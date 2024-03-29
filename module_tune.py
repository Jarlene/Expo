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
from layer.ssm import Mamba
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

    quantizer: Optional[bool] = field(default=True,
                                      metadata={
                                          "help": "load model quantizer"})
    max_length: Optional[int] = field(
        default=8192, metadata={"help": "max length for model input"})

    d_state: Optional[int] = field(
        default=16,
        metadata={"help": "ssm model d_state"}
    )
    d_conv: Optional[int] = field(
        default=4,
        metadata={"help": "ssm model d_conv"}
    )
    expand: Optional[int] = field(
        default=2,
        metadata={"help": "ssm model expand"}
    )

    beta: Optional[float] = field(
        default=1.0,
        metadata={"help": "ssm model expand"}
    )

    parallel: Optional[bool] = field(
        default=True,
        metadata={"help": "ssm model parallel expand"}
    )
    split_num: Optional[int] = field(
        default=4,
        metadata={"help": "split_num"}
    )
    padding_side: Optional[str] = field(
        default='left', metadata={'help': "tokenizer padding side"})


def get_tokenizer(script_args: ScriptArguments):
    need_resize_embed = False
    if script_args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.tokenizer_path, padding_side=script_args.padding_side, trust_remote_code=True)
        need_resize_embed = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name_or_path, padding_side=script_args.padding_side, trust_remote_code=True)
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


def generate_prompt_v1(data):
    template = """标题：{title}
内容类别：{dataType}
内容：{content}"""
    return template.format(title=data['title'], dataType=data['dataType'], content=data['content'])


class ModuleHook(torch.nn.Module):

    def __init__(self, base_layer: torch.nn.Module, hidden_size, is_parallel=True, target: torch.nn.Module = None) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.target = target
        if self.target is None:
            self.target = copy.deepcopy(base_layer)
        self.target.requires_grad_(True)
        self.base_layer.requires_grad_(False)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.is_parallel = is_parallel
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if not self.is_parallel:
            if type(self.target) == type(self.base_layer):
                hidden_states = self.target(hidden_states)[0]*self.beta
            else:
                hidden_states = self.target(
                    self.norm(hidden_states)) * self.beta
            result = self.base_layer(hidden_states=hidden_states, attention_mask=attention_mask,
                                     past_key_value=past_key_value,
                                     output_attentions=output_attentions, use_cache=use_cache, **kwargs)
        else:
            if type(self.target) == type(self.base_layer):
                res = self.target(hidden_states)[0]*self.beta
            else:
                res = self.target(self.norm(hidden_states)) * self.beta
            result = self.base_layer(hidden_states=hidden_states, attention_mask=attention_mask,
                                     past_key_value=past_key_value,
                                     output_attentions=output_attentions, use_cache=use_cache, **kwargs)
            ss = list(result)
            ss[0] += res
            result = tuple(ss)
        return result


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
    target = Mamba(dim=model.config.hidden_size, d_conv=script_args.d_conv,
                   d_state=script_args.d_state, expand=script_args.expand)
    for idx, child in enumerate(model.model.layers):
        if idx % script_args.split_num == 0:
            new_child = ModuleHook(
                child, model.config.hidden_size, target=target, is_parallel=script_args.parallel)
        else:
            new_child = child
        model.model.layers[idx] = new_child
    print(model)
    model.config.d_state = script_args.d_state
    model.config.d_conv = script_args.d_conv
    model.config.expand = script_args.expand
    model.config.is_parallel = script_args.parallel
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

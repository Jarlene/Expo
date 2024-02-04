from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
import copy
import torch
import torch.nn as nn
from datasets import load_from_disk, DatasetDict
from peft import MoELoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, DefaultDataCollator, BitsAndBytesConfig, Trainer
from sklearn.metrics import accuracy_score
from utils import get_train_args
from trainer import get_trainer


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
    template = """标题：{title}
内容类别：{dataType}
内容：{content}"""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        contents = [self.template.format(
            title=d['title'], dataType=d['dataType'], content=d['content']) for d in features]
        output = self.tokenizer(contents,
                                padding='longest',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='pt')
        output['labels'] = output['input_ids'].clone()
        return output


@dataclass
class ScriptArguments(TrainingArguments):
    """
    The arguments for the sft training script.
    """

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )

    quantizer: Optional[bool] = field(default=True,
                                      metadata={
                                          "help": "load model quantizer"})
    num_experts: Optional[int] = field(
        default=4, metadata={"help": "the num of experts"})
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


def get_data(script_args: ScriptArguments):
    raw_data = load_from_disk(script_args.data_dir)['train']
    data = raw_data.train_test_split(
        test_size=script_args.val_data_percentage)
    return data['train'], data['test']


def get_model_and_tokenizer(script_args: ScriptArguments, training_args: TrainingArguments, trainable=False):
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
    if trainable:
        model.config.use_cache = False
    peft_config = get_lora_config(script_args)
    if training_args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(
            model, training_args.resume_from_checkpoint)
    else:
        model = get_peft_model(model, peft_config, adapter_name='sft')
    model.print_trainable_parameters()
    print(model)
    tokenizer, need_resize_embed = get_tokenizer(script_args)
    if need_resize_embed:
        model.resize_token_embeddings(len(tokenizer))
    if script_args.num_experts > 0:
        model.config.num_experts = script_args.num_experts
        model.config.router_jitter_noise = script_args.router_jitter_noise
        model.config.num_experts_per_token = script_args.num_experts_per_token
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tokenizer


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
        collate_fn=data_collator,
        compute_metrics=compute_metrics,
    )

    resume_from_checkpoint = script_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model(output_dir=script_args.output_dir + "/result")
    trainer.model.save_pretrained(
        script_args.output_dir + "/sft", safe_serialization=True)
    tokenizer.save_pretrained(script_args.output_dir + "/sft")


if __name__ == "__main__":
    main()

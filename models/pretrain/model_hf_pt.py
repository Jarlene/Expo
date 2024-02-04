from trainer.trainer import get_trainer
from models.pretrain.blink.modeling_blink import BlinkForCausalLM
from models.pretrain.blink.configuration_blink import BlinkConfig
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, DefaultDataCollator, TrainingArguments
from utils.utils import get_train_args
import torch
import math
from dataclasses import dataclass, field


@dataclass
class ScriptArguments(TrainingArguments):
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "train test split ratio"})

    data_dir: str = field(
        default=None,
        metadata={"help": "data dir "})

    val_data_percentage: Optional[float] = field(
        default=0.00001,
        metadata={"help": "train test split ratio"})

    hidden_size: int = field(
        default=2048,
        metadata={"help": "model hidden size"})

    intermediate_size: int = field(
        default=2048*4,
        metadata={"help": "model ffn intermediate size"})

    num_hidden_layers: int = field(
        default=12,
        metadata={"help": "model num of hidden layers"})

    num_attention_heads: int = field(
        default=8,
        metadata={"help": "model num of attention heads"})

    num_key_value_heads: int = field(
        default=2,
        metadata={"help": "model nnum_key_value_heads"})

    max_length: int = field(
        default=4096,
        metadata={"help": "model nnum_key_value_heads"})

    sliding_window_size: Optional[int] = field(
        default=None,
        metadata={"help": "attention sliding window size"})

    use_moe: bool = field(
        default=False,
        metadata={"help": "use moe or not"}
    )
    use_soft_moe: bool = field(
        default=False,
        metadata={"help": "use soft moe or not"}
    )
    moe_num_experts: int = field(
        default=4,
        metadata={"help": "moe num experts"}
    )
    moe_num_slots: int = field(
        default=32,
        metadata={"help": "moe num slots"}
    )


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


def get_tokenizer(script_args: ScriptArguments) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def get_data(script_args: ScriptArguments) -> Tuple[Dataset, Dataset]:
    raw_data = load_from_disk(script_args.data_dir)['train']
    data = raw_data.train_test_split(
        test_size=script_args.val_data_percentage)
    return data['train'], data['test']


def get_model_and_tokenizer(script_args: ScriptArguments) -> Tuple[BlinkForCausalLM, AutoTokenizer, BlinkConfig]:
    tokenizer = get_tokenizer(script_args)
    config = BlinkConfig(vocab_size=len(tokenizer),
                         hidden_size=script_args.hidden_size,
                         intermediate_size=script_args.intermediate_size,
                         num_hidden_layers=script_args.num_hidden_layers,
                         num_attention_heads=script_args.num_attention_heads,
                         num_key_value_heads=script_args.num_key_value_heads,
                         max_position_embeddings=script_args.max_length,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id,
                         bos_token_id=tokenizer.bos_token_id,
                         sliding_window=script_args.sliding_window_size,
                         use_moe=script_args.use_moe,
                         moe_soft=script_args.use_soft_moe,
                         moe_num_experts=script_args.moe_num_experts,
                         torch_dtype=torch.bfloat16,
                         moe_num_slots=script_args.moe_num_slots,
                         )
    model = BlinkForCausalLM(config=config)
    return model, tokenizer, config


def main():
    script_args = get_train_args(ScriptArguments)
    train_dataset, eval_dataset = get_data(script_args)
    model, tokenizer, config = get_model_and_tokenizer(script_args)
    data_collator = HugeDataCollator(
        tokenizer=tokenizer, max_length=config.max_position_embeddings)
    trainer = get_trainer(
        args=script_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        tokenizer=tokenizer,
        collate_fn=data_collator)
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
        script_args.output_dir + "/pt", safe_serialization=True)
    tokenizer.save_pretrained(script_args.output_dir + "/pt")


if __name__ == "__main__":
    main()

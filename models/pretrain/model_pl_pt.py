from trainer.trainer import get_trainer
from models.pretrain.blink.modeling_blink import BlinkForCausalLM
from models.pretrain.blink.configuration_blink import BlinkConfig
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, DefaultDataCollator
from utils.utils import TrainArguments, get_train_args
import torch


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


def get_tokenizer(script_args: TrainArguments) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def get_data(script_args: TrainArguments) -> Tuple[Dataset, Dataset]:
    raw_data = load_from_disk(script_args.data_dir)['train']
    data = raw_data.train_test_split(
        test_size=script_args.val_data_percentage)
    return data['train'], data['test']


def get_model_and_tokenizer(script_args: TrainArguments) -> Tuple[BlinkForCausalLM, AutoTokenizer, BlinkConfig]:
    tokenizer = get_tokenizer(script_args)
    config = BlinkConfig(vocab_size=len(tokenizer),
                         hidden_size=2048,
                         intermediate_size=2048*4,
                         num_hidden_layers=8,
                         num_attention_heads=8,
                         num_key_value_heads=2,
                         max_position_embeddings=2048,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id,
                         bos_token_id=tokenizer.bos_token_id,
                         sliding_window=512,
                         use_moe=True,
                         moe_soft=False,
                         moe_num_experts=8,
                         torch_dtype=torch.bfloat16,
                         moe_num_slots=32,
                         )
    model = BlinkForCausalLM(config=config)
    return model, tokenizer, config


def main():
    script_args = get_train_args()
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
    trainer.train()
    trainer.save_pretrained()


if __name__ == "__main__":
    main()

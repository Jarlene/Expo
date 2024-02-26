import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from utils.utils import TrainArguments, get_train_args
from trainer.trainer import get_trainer
from models.pretrain.blink.modeling_blink import BlinkForCausalLM, BlinkForSequenceClassification
from models.pretrain.blink.configuration_blink import BlinkConfig
from models.pretrain.ssmformer.configuration_ssmformer import SSMFormerConfig
from models.pretrain.ssmformer.modeling_ssmformer import SSMFormerForCausalLM, SSMFormerForSequenceClassification
from models.pretrain.moduleformer.configuration_moduleformer import ModuleFormerConfig
from models.pretrain.moduleformer.modeling_moduleformer import ModuleFormerForCausalLM, ModuleFormerForSequenceClassification
from metric.acc import Accuracy
from torchmetrics.text.perplexity import Perplexity

AutoConfig.register("blink", BlinkConfig)
AutoConfig.register("ssmformer", SSMFormerConfig)
AutoConfig.register("moduleformer", ModuleFormerConfig)
AutoModelForCausalLM.register(BlinkConfig, BlinkForCausalLM)
AutoModelForCausalLM.register(SSMFormerConfig, SSMFormerForCausalLM)
AutoModelForCausalLM.register(ModuleFormerConfig, ModuleFormerForCausalLM)
AutoModelForSequenceClassification.register(
    BlinkConfig, BlinkForSequenceClassification)
AutoModelForSequenceClassification.register(
    SSMFormerConfig, SSMFormerForSequenceClassification)
AutoModelForSequenceClassification.register(
    ModuleFormerConfig, ModuleFormerForSequenceClassification)


def generate_prompt(data):
    template = """标题：{title}
内容类别：{dataType}
内容：{content}"""
    return template.format(title=data['title'], dataType=data['dataType'], content=data['content'])


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
        default=hidden_size.real*4,
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


def get_model_and_tokenizer(script_args: ScriptArguments) -> Tuple[SSMFormerForCausalLM, AutoTokenizer, SSMFormerConfig]:
    tokenizer = get_tokenizer(script_args)
    if script_args.model_name_or_path == 'blink':
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
                             moe_num_slots=script_args.moe_num_slots,

                             )
    elif script_args.model_name_or_path == 'ssmformer':
        config = SSMFormerConfig(
            vocab_size=len(tokenizer),
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
            moe_num_slots=script_args.moe_num_slots,
            d_state=script_args.d_state,
            d_conv=script_args.d_conv,
            expand=script_args.expand,
        )
    elif script_args.model_name_or_path == 'moduleformer':
        config = ModuleFormerConfig(
            vocab_size=len(tokenizer),
            hidden_size=script_args.hidden_size,
            max_position_embeddings=script_args.max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    else:
        config = AutoConfig.from_pretrained(
            script_args.model_name_or_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_config(
        config=config, trust_remote_code=True)
    return model, tokenizer, config


def main():
    script_args = get_train_args(ScriptArguments)
    train_dataset, eval_dataset = get_data(script_args)
    model, tokenizer, config = get_model_and_tokenizer(script_args)
    data_collator = HugeDataCollator(
        tokenizer=tokenizer, max_length=config.max_position_embeddings, generate_func=generate_prompt)
    trainer = get_trainer(
        args=script_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        tokenizer=tokenizer,
        collate_fn=data_collator,
        metrics=[Accuracy(), Perplexity(ignore_index=-100)])
    trainer.train()
    trainer.save_pretrained()


if __name__ == "__main__":
    main()

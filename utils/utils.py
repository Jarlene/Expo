from argparse import Namespace
from dataclasses import dataclass, field
from typing import Union, Optional, TypeVar
from transformers import HfArgumentParser
import torch
T = TypeVar('T')


@dataclass
class TrainArguments(Namespace):
    data_dir: str = field(
        default=None,
        metadata={"help": "data dir "},
    )
    batch_size: Optional[int] = field(
        default=2,
        metadata={"help": "train batch size"})

    step_size: Optional[int] = field(
        default=-1,
        metadata={"help": "train step size"})

    lr: Optional[float] = field(
        default=0.00001,
        metadata={"help": "train learning rate"})

    script_able: bool = field(
        default=False,
        metadata={"help": "convert model to torchscript"})

    weight_decay: float = field(default=0.01, metadata={
                                "help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={
                              "help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={
                              "help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    resume: Optional[bool] = field(
        default=False,
        metadata={"help": "resume training from checkpoint"})

    output_dir: Optional[str] = field(
        default='save',
        metadata={"help": "output dir"})

    devices: Optional[str] = field(
        default='auto',
        metadata={"help": "use lightning training model devices"})

    strategy: Optional[str] = field(
        default='auto',
        metadata={"help": "use lightning training model strategy"})
    
    num_nodes: Optional[int] = field(
        default=1,
        metadata={"help": "use lightning training model num nodes"})

    precision: Optional[str] = field(
        default='bf16',
        metadata={"help": "use lightning training model precision"})

    log_dir: Optional[str] = field(
        default='logs',
        metadata={"help": "log dir"})

    name: Optional[str] = field(
        default='tensorboard',
        metadata={"help": "experiment name"})

    version: Optional[str] = field(
        default='v1',
        metadata={"help": "experiment version"})

    monitor:  Optional[str] = field(
        default='val_loss',
        metadata={"help": "monitor metric"})

    num_epochs: Optional[int] = field(
        default=-1,
        metadata={"help": "train epochs"})
    log_every_n_steps: Optional[int] = field(
        default=10,
        metadata={"help": "How often to log within steps."})

    val_check_interval: Optional[int] = field(
        default=200, metadata={"help": "How often to check the validation set"})

    num_proc: Optional[int] = field(
        default=8,
        metadata={"help": "DataLoader num workers"})

    pin_memory: Optional[bool] = field(
        default=True,
        metadata={"help": "pin memory"})

    seed: Optional[int] = field(
        default=42,
        metadata={"help": "seed"})

    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "gradient accumulation steps"})

    val_data_percentage: Optional[float] = field(
        default=0.00001,
        metadata={"help": "train test split ratio"})
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "tokenizer path"})
    model_name_or_path:str = field(
        default=None,
        metadata={"help": "the location of the model name or path"},
    )
    quantizer: Optional[bool] = field(default=None,
                                      metadata={
                                          "help": "load model quantizer"})

def get_train_args(clazz: Optional[T] = None) -> Union[TrainArguments, T]:
    if clazz is None:
        clazz = (TrainArguments)
    elif not isinstance(clazz, tuple):
        clazz = (clazz)
    parser = HfArgumentParser(clazz)
    return parser.parse_args_into_dataclasses()[0]


def set_module_requires_grad(
        module: torch.nn.Module,
        requires_grad: bool):
    module.requires_grad_(requires_grad)

import os
import argparse
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Union, Optional, Dict, List, Tuple, TypeVar
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader


from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins import TorchCheckpointIO

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    DeviceStatsMonitor,

)
import transformers
from transformers import TrainingArguments, PreTrainedModel, AutoTokenizer
from peft.peft_model import PeftModel
from models.base import Base
from utils.utils import TrainArguments


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


class PLTrainer(Trainer):
    def __init__(self, args: TrainArguments,
                 model: nn.Module | PreTrainedModel | PeftModel,
                 train_dataset: Dataset,
                 eval_dataset: Dataset = None,
                 collate_fn=None,
                 tokenizer: AutoTokenizer = None,
                 example_input_array=None, **kwargs):
        super().__init__(**kwargs)
        seed_worker()
        self.args = args
        self.resume = args.resume
        self.training_model = Base(model, args)
        self.tokenizer = tokenizer
        if example_input_array is None:
            example_input_array = collate_fn(
                [train_dataset[0]]) if collate_fn is not None else train_dataset[0]
            if isinstance(example_input_array, list) or isinstance(example_input_array, tuple):
                example_input_array = example_input_array[0]

        examples = {}
        for k, v in example_input_array.items():
            examples[k] = v.to(self.training_model.device)
        self.training_model.example_input_array = examples
        if eval_dataset is None:
            total_length = len(train_dataset)
            valid_length = int(total_length * args.val_data_percentage)
            train_length = total_length - valid_length
            train_dataset, eval_dataset = random_split(
                train_dataset, (train_length, valid_length))

        self.training_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_proc,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
            shuffle=True,
            worker_init_fn=seed_worker,
        )
        self.validate_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_proc,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def train(self):
        super().fit(model=self.training_model,
                    train_dataloaders=self.training_dataloader,
                    val_dataloaders=self.validate_dataloader,
                    ckpt_path='last' if self.resume else None)

    def save_pretrained(self):
        self.training_model.save_pretrained(self.args.output_dir + "/result")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.args.output_dir + "/result")

    def to_torchscript(self, ckpt_path):
        ckpt_path = self._checkpoint_connector._select_ckpt_path(
            self.state.fn,
            ckpt_path,
            model_provided=True,
            model_connected=self.training_model is not None,
        )
        self.training_model.load_from_checkpoint(ckpt_path)
        self.training_model.to_torchscript(self.args.output_dir + "/model.pt")


class HFTrainer(transformers.Trainer):
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module, PeftModel],
                 args: TrainingArguments,
                 train_dataset: Dataset,
                 eval_dataset: Dataset = None,
                 collate_fn: Any | None = None, **kwargs):
        super().__init__(model=model, args=args, data_collator=collate_fn,
                         train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)
        seed_worker()
        self.hf_model = isinstance(
            model, PreTrainedModel) or isinstance(model, PeftModel)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.hf_model:
            return super().compute_loss(model, inputs, return_outputs)
        else:
            logists = model(**inputs)
            if hasattr(model, "module"):
                loss_module = model.module
            else:
                loss_module = model

            if hasattr(loss_module, 'compute_loss'):
                loss = loss_module.compute_loss(**inputs, **logists)
            else:
                RuntimeError(
                    "compute_loss function must in {}".format(type(loss_module)))
            return loss if not return_outputs else (loss, logists)

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None,
              trial: Union["optuna.Trial", Dict[str, Any]] = None,
              ignore_keys_for_eval: Optional[List[str]] = None,
              **kwargs):
        if resume_from_checkpoint is None and self.args.resume_from_checkpoint is not None:
            resume_from_checkpoint = self.args.resume_from_checkpoint
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def save_pretrained(self):
        self.save_model(self.args.output_dir + "/result")


def get_trainer(args: TrainArguments | TrainingArguments,
                model: nn.Module,
                train_dataset: Dataset,
                eval_dataset: Dataset = None,
                collate_fn=None,
                tokenizer: AutoTokenizer = None,
                example_input_array=None, **kwargs) -> Union[PLTrainer, HFTrainer]:
    if isinstance(args, TrainArguments):
        trainer = PLTrainer(args=args,
                            model=model,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            tokenizer=tokenizer,
                            collate_fn=collate_fn,
                            example_input_array=example_input_array,
                            log_every_n_steps=args.log_every_n_steps,
                            val_check_interval=args.val_check_interval,
                            logger=TensorBoardLogger(
                                save_dir=args.log_dir,
                                log_graph=True,
                                name=args.name,
                                version=args.version),
                            strategy=args.strategy,
                            devices=args.devices,
                            precision=args.precision,
                            num_nodes=args.num_epochs,
                            enable_checkpointing=True,
                            callbacks=[
                                LearningRateMonitor(),
                                ModelCheckpoint(save_top_k=2,
                                                dirpath=os.path.join(args.output_dir,
                                                                     args.name,
                                                                     args.version),
                                                monitor=args.monitor,
                                                every_n_train_steps=args.val_check_interval,
                                                save_last=True),
                                # EarlyStopping(monitor=args.monitor),
                                # GradientAccumulationScheduler(scheduling={2: 1}),
                                RichProgressBar(leave=True),
                                RichModelSummary(3),
                                DeviceStatsMonitor(),
                            ],
                            plugins=[TorchCheckpointIO()],
                            #    profiler=PyTorchProfiler(),
                            max_epochs=args.num_epochs,
                            max_steps=args.step_size,
                            **kwargs)
    else:
        trainer = HFTrainer(model=model, args=args,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            collate_fn=collate_fn,
                            tokenizer=tokenizer, **kwargs)
    return trainer

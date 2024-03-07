
import inspect
from typing import Union, Any, Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import PreTrainedModel, AutoTokenizer
from peft.peft_model import PeftModel
from lightning.pytorch.core import LightningModule
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from torchmetrics import Metric
from utils.utils import TrainArguments
from dataclasses import asdict


class Base(LightningModule):

    def __init__(self,
                 model: Union[nn.Module, PreTrainedModel, PeftModel],
                 tokenizer: AutoTokenizer,
                 args: TrainArguments,
                 metrics: Optional[Union[Metric, List[Metric]]] = None, **kwargs) -> None:
        super(Base, self).__init__(**kwargs)
        self.save_hyperparameters(asdict(args))
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.metrics = []
        self.hf_model = isinstance(
            model, PreTrainedModel) or isinstance(model, PeftModel)
        self.register_metrics(metrics)
        self.__jit_unused_properties__.append("tokenizer")
        self.__jit_unused_properties__.append("metrics")
        self.__jit_unused_properties__.append("args")

    def register_metrics(self, metrics: Optional[Union[Metric, List[Metric]]] = None):
        if metrics is None:
            return
        if isinstance(metrics, Metric):
            metrics = [metrics]
        for m in self.metrics:
            self.metrics.append(m.to(self.device))

    def generate(self, *args, **kwargs):
        if self.hf_model:
            return self.model.generate(*args, **kwargs)

    def configure_model(self):
        self.print("configure_model")
        for m in self.metrics:
            m.to(self.device)
        self.model.to(self.device)
        if self.example_input_array:
            self.example_input_array = self.prepare_inputs(
                self.example_input_array)

    def prepare_inputs(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Dict):
            return type(data)({k: self.prepare_inputs(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self.prepare_inputs(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.as_tensor(data).to(self.device)
        else:
            return data

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_loss(self, inputs, logists, stage='train', prog_bar=True):
        if self.hf_model:
            loss = logists.loss
        elif hasattr(self.model, "compute_loss"):
            loss = self.model.compute_loss(**inputs, **logists)
        else:
            loss = logists['loss']
        self.log(name=stage + '_loss', value=loss,
                 sync_dist=True, prog_bar=prog_bar, logger=True)
        return loss

    def compute_metrics(self, inputs, logists, stage='train'):
        if hasattr(self.model, 'compute_metrics'):
            metrics_val = self.model.compute_metrics(**inputs, **logists)
            for k, v in metrics_val.items():
                self.log(name=stage + '_' + k, value=v,
                         sync_dist=True, logger=True)
        else:
            for m in self.metrics:
                res = m(**logists, **inputs)
                self.log(name=stage + '_' + m.__class__.__name__, value=res,
                         sync_dist=True, logger=True)

    def reset_metric(self):
        if hasattr(self.model, 'metrics_reset'):
            self.model.metrics_reset()
        else:
            for m in self.metrics:
                m.reset()

    def training_step(self, batch, batch_idx):

        if not isinstance(batch, dict):
            RuntimeError(
                "model input must be dict and key is model forward params")

        inputs = self.prepare_inputs(batch)
        logists = self.forward(**inputs)
        self.compute_metrics(inputs, logists)
        loss = self.compute_loss(inputs, logists)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.prepare_inputs(batch)
        logists = self.forward(**inputs)
        self.compute_metrics(inputs, logists, stage='val')
        loss = self.compute_loss(inputs, logists, stage='val', prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):

        inputs = self.prepare_inputs(batch)
        logists = self.forward(**inputs)
        self.compute_metrics(inputs, logists, stage='test')
        loss = self.compute_loss(inputs, logists, stage='test', prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        self.reset_metric()

    def on_validation_epoch_end(self):
        self.reset_metric()

    def on_test_epoch_end(self):
        self.reset_metric()

    def configure_optimizers(self):
        opt_class = AdamW
        if 'deepspeed' in self.args.strategy:
            opt_class = FusedAdam
            if 'offload' in self.args.strategy:
                opt_class = DeepSpeedCPUAdam
        optim = opt_class(self.trainer.model.parameters(),
                          lr=self.args.lr,
                          weight_decay=self.args.weight_decay,
                          betas=(self.args.adam_beta1, self.args.adam_beta2))
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optim),
                "monitor": 'val_loss',
            },
        }

    def save_pretrained(self, path):
        if self.hf_model:
            self.model.save_pretrained(path, safe_serialization=False)
            if self.tokenizer:
                self.tokenizer.save_pretrained(path)


class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for autoregressive that returns a scalar for each output token.
    The weights of the value head need to be in FP32.
    """

    def __init__(self, hidden_size: int, dropout_prob: float = 0.3, ** kwargs):
        super().__init__()

        self.summary = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        # detach so that loss isn't backproped through LM
        # upcast since fp32 is important for good value predictions
        hidden_states = hidden_states.detach().to(torch.float32)
        output = self.summary(hidden_states)
        return output

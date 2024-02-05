
import inspect
from typing import Union, Any, Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from models.base import Base
from .slic import SLiCModel
from transformers import PreTrainedModel, AutoTokenizer
from peft.peft_model import PeftModel
from utils.utils import TrainArguments
from torchmetrics import Metric


class SPINModel(SLiCModel):
    def __init__(self,
                 model: Union[nn.Module, PreTrainedModel, PeftModel],
                 tokenizer: AutoTokenizer,
                 args: TrainArguments,
                 num_iterations: int = 3,
                 beta: float = 0.3,
                 metrics: Optional[Union[Metric, List[Metric]]] = None, **kwargs):
        super().__init__(model, tokenizer, args, metrics, **kwargs)
        self.num_iterations = num_iterations
        self.beta = beta

    def on_train_epoch_start(self):
        self.prepare_data()

    def compute_loss(self, batch):
        labels_hat = batch.pop('labels_hat')
        labels = batch.pop('labels')

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        return loss

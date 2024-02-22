from typing import Union, Any, Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from models.alignment.unpaired import UnpairedPreferenceModel
from models.alignment.paired import PairedPreferenceModel
from transformers import PreTrainedModel, AutoTokenizer
from peft.peft_model import PeftModel
from utils.utils import TrainArguments
from torchmetrics import Metric
from torchmetrics.text.bert import BERTScore


class SLiCModel(UnpairedPreferenceModel):
    def __init__(self,
                 model: Union[nn.Module, PreTrainedModel, PeftModel],
                 tokenizer: AutoTokenizer,
                 args: TrainArguments,
                 reference_model: Optional[Union[nn.Module,
                                                 PreTrainedModel]] = None,
                 metrics: Optional[Union[Metric, List[Metric]]] = None, **kwargs) -> None:
        super().__init__(model, tokenizer, args, reference_model, metrics, **kwargs)

    def similarty(self, batch):
        emb_hat_y = self.model.get_input_embeddings()(batch['labels_hat'])
        emb_y = self.model.get_input_embeddings()(batch['labels'])

    def prepare_data(self):
        dataloader = self.train_dataloader()
        # Generate synthetic data
        labels = None
        for batch_idx,  batch in enumerate(dataloader):
            if 'labels_hat' in batch.keys():
                batch.pop('labels_hat')
            with torch.no_grad():
                res = self.model.generate(**batch)
            batch["labels_hat"] = res


    def on_train_epoch_start(self):
        self.prepare_data()

    def compute_loss(self, batch):
        labels_hat = batch.pop('labels_hat')
        labels = batch.pop('labels')

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        return loss


class SLiCHFModel(PairedPreferenceModel):
    def __init__(self,
                 model: Union[nn.Module, PreTrainedModel, PeftModel],
                 tokenizer: AutoTokenizer,
                 args: TrainArguments,
                 reference_model: Optional[Union[nn.Module,
                                                 PreTrainedModel]] = None,
                 metrics: Optional[Union[Metric, List[Metric]]] = None, **kwargs) -> None:
        super().__init__(model, tokenizer, args, reference_model, metrics, **kwargs)

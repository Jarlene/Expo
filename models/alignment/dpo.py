from typing import Union, Any, Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from models.alignment.paired import PairedPreferenceModel
from transformers import PreTrainedModel, AutoTokenizer
from peft.peft_model import PeftModel
from utils.utils import TrainArguments
from torchmetrics import Metric


class DpoModel(PairedPreferenceModel):
    def __init__(self,
                 model: Union[nn.Module, PreTrainedModel, PeftModel],
                 tokenizer: AutoTokenizer,
                 args: TrainArguments,
                 reference_model: Optional[Union[nn.Module,
                                                 PreTrainedModel]] = None,
                 metrics: Optional[Union[Metric, List[Metric]]] = None, **kwargs) -> None:
        super().__init__(model, tokenizer, args, reference_model, metrics, **kwargs)


class CDDpoModel(PairedPreferenceModel):
    def __init__(self,
                 model: Union[nn.Module, PreTrainedModel, PeftModel],
                 tokenizer: AutoTokenizer,
                 args: TrainArguments,
                 reference_model: Optional[Union[nn.Module,
                                                 PreTrainedModel]] = None,
                 metrics: Optional[Union[Metric, List[Metric]]] = None, **kwargs) -> None:
        super().__init__(model, tokenizer, args, reference_model, metrics, **kwargs)

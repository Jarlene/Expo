from typing import Union, Any, Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from models.base import Base, ValueHead
from transformers import PreTrainedModel, AutoTokenizer
from peft.peft_model import PeftModel
from utils.utils import TrainArguments
from torchmetrics import Metric


class PairedPreferenceModel(Base):
    def __init__(self,
                 model: Union[nn.Module, PreTrainedModel, PeftModel],
                 tokenizer: AutoTokenizer,
                 args: TrainArguments,
                 reference_model: Optional[Union[nn.Module,
                                                 PreTrainedModel]] = None,
                 metrics: Optional[Union[Metric, List[Metric]]] = None,
                 **kwargs) -> None:
        super().__init__(model, tokenizer, args, metrics, **kwargs)
        self.reference_model = reference_model
        self.ref_hf_model = reference_model and isinstance(
            reference_model, PreTrainedModel)

    def sample(self, include_original_prompt=False) -> List[Dict[str, str]]:
        """
        Generate samples from the policy model.

        Args:
            include_original_prompt: whether to include the original prompt among the generated text

        Returns:
            A list of samples, each of which is of the form:
            {
                'prompt': the input
                'chosen': the generation chosen by the human for the given prompt
                'policy': the generation from the policy model
            }
        """
        all_policy_samples, all_prompts, all_chosen, all_original_prompts = [], [], [], []
        samples = []
        self.model.eval()
        if self.reference_model is not None:
            self.reference_model.eval()

        for eval_batch in self.val_dataloader():
            policy_samples = self.get_batch_samples(eval_batch)

            chosen_samples = []
            # for DPO-like losses, chosen_text is the field that will contain the text; target_text for all other losses
            # be sure to remove EOS token if present
            for x in (eval_batch['target_text'] if 'target_text' in eval_batch else eval_batch['chosen_text']):
                if self.tokenizer.eos_token in x:
                    x = x[:x.rfind(self.tokenizer.eos_token)]

                chosen_samples.append(x)

            all_prompts.extend(eval_batch['prompt_text'])
            all_original_prompts.extend(eval_batch['original_prompt'])
            all_chosen.extend(chosen_samples)
            all_policy_samples.extend(policy_samples)

            samples_num = getattr(self.args, 'samples_num')
            if samples_num is not None and len(all_prompts) > samples_num:
                break
            else:
                self.print(f"Generated {len(all_prompts)} samples ...")

        for i in range(len(all_prompts)):
            if include_original_prompt:
                samples.append({
                    'prompt': all_prompts[i],
                    'chosen': all_chosen[i],
                    'policy': all_policy_samples[i][len(all_prompts[i]):],
                    'original_prompt': all_original_prompts[i],
                })
            else:
                samples.append({
                    'prompt': all_prompts[i],
                    'chosen': all_chosen[i],
                    'policy': all_policy_samples[i][len(all_prompts[i]):],
                })

        return samples

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """
        Generate samples from the policy model for a given batch.

        Args:
            batch: a batch of data

        Returns:
            A list of samples, each of which is a string
        """
        raise NotImplementedError

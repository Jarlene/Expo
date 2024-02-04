from torchmetrics import Metric
import torch
import torch.nn.functional as F


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        # to count the correct predictions
        self.add_state('corrects', default=torch.tensor(0))
        # to count the total predictions
        self.add_state('total', default=torch.tensor(0))

    def update(self, preds, target):
        # update correct predictions count
        preds = torch.argmax(F.softmax(preds, dim=-1), dim=-1)
        self.corrects += torch.sum(preds == target)
        # update total count, numel() returns the total number of elements
        self.total += target.numel()

    def compute(self):
        # final computation
        return self.corrects / self.total

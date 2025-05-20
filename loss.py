import torch
from torch import nn
class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, vocab_size, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.vocab_size = vocab_size

    def forward(self, pred, target):
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  
        ignore = target == self.ignore_index
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        pred = pred.log_softmax(dim=-1)
        loss = self.criterion(pred, true_dist)
        return loss / (~ignore).sum()

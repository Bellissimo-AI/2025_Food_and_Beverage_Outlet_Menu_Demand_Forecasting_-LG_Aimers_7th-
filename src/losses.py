
import torch
import torch.nn as nn

class ScaledSMAPELoss(nn.Module):
    def __init__(self, zero_weight=0.3):
        super().__init__()
        self.zero_weight = zero_weight
        self.epsilon = 1e-8

    def forward(self, y_pred, y_true):
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0 + self.epsilon
        smapes = (numerator / denominator) * 100
        value_weights = torch.where(y_true == 0, self.zero_weight, 1.0)
        return torch.mean(smapes * value_weights)

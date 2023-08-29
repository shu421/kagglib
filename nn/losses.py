import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=0)
        return torch.mean(torch.sqrt(colwise_mse), dim=0)

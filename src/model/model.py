import torch.nn as nn
import torch as torch
from .transformer import TransformerBlock
from .moe import SSMoE

class EEGMoEModel(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.encoder = nn.Linear(
            input_dim,
            64
        )

        self.transformer = TransformerBlock(
            64
        )

        self.moe = SSMoE(
            64
        )

        self.classifier = nn.Linear(
            64,
            1
        )

    def forward(self, x):

     x = self.encoder(x)

     x = x.unsqueeze(1)

     x = self.transformer(x)

     x = self.moe(x)

     x = self.classifier(x)

    # FIX: convert to probability
    

     return x
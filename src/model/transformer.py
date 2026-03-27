import torch.nn as nn

class TransformerBlock(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=4,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        out, _ = self.attn(
            x,
            x,
            x
        )

        return self.norm(
            x + out
        )
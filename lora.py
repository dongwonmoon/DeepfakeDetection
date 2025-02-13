import math
import torch
import torch.nn as nn


# Define a LoRA adapter for Linear layers
class LoRALinear(nn.Module):
    def __init__(
        self, in_features, out_features, r=4, alpha=32, dropout=0.0, bias=True
    ):
        """
        LoRALinear wraps a frozen pretrained weight with trainable low-rank adapters.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            r (int): Rank of the low-rank decomposition.
            alpha (int): Scaling factor.
            dropout (float): Dropout rate applied to inputs.
            bias (bool): Whether to include bias.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        # The original weight matrix is frozen
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.requires_grad = False  # Freeze original weight
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None

        # Low-rank adaptation matrices, trainable
        self.A = nn.Parameter(torch.Tensor(r, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize low-rank matrices; original weight should be loaded from pretrained model.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Original frozen part
        base = nn.functional.linear(x, self.weight, self.bias)
        # LoRA adaptation
        lora_update = self.dropout(x) @ self.A.t()  # shape: (batch, r)
        lora_update = lora_update @ self.B.t()  # shape: (batch, out_features)
        return base + self.scaling * lora_update

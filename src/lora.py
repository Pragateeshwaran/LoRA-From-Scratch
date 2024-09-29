# lora.py

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class LoRA_Scratch(nn.Module):
    """
    Implements Low-Rank Adaptation (LoRA) for a given linear layer.
    """
    def __init__(self, layer, rank, alpha=0.5):
        super(LoRA_Scratch, self).__init__()
        feature_in, feature_out = layer.weight.shape

        # Initialize A and B with zeros
        self.A = nn.Parameter(torch.zeros(feature_in, rank).to(device=device))
        self.B = nn.Parameter(torch.zeros(rank, feature_out).to(device=device))
        
        self.scale = alpha / rank
        self.LoRA = True  # Flag to enable/disable LoRA

    def forward(self, X):
        """
        Forward pass with LoRA adaptation.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Adapted output tensor.
        """
        if self.LoRA:
            return X + (self.A @ self.B) * self.scale
        else:
            return X

def LoRA_convertor(layer, rank=4, lora_alpha=2):
    """
    Factory function to create a LoRA_Scratch instance for a given layer.

    Args:
        layer (nn.Module): The linear layer to apply LoRA to.
        rank (int): Rank of the low-rank matrices A and B.
        lora_alpha (float): Scaling factor.

    Returns:
        LoRA_Scratch: An instance of LoRA_Scratch.
    """
    return LoRA_Scratch(layer, rank=rank, alpha=lora_alpha)

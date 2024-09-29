import torch.nn.utils.parametrize as parametrize
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

    def LoRA_convertor(self, layer, rank=4, lora_alpha=2):
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


    def apply_lora(self, model, rank=4, lora_alpha=2):
        """
        Applies LoRA parametrization to all linear layers in the model.

        Args:
            model (nn.Module): The neural network model.
            rank (int): Rank of the LoRA matrices.
            lora_alpha (float): Scaling factor for LoRA.
        """
        for layer in [model.layer1, model.layer2, model.layer3]:
            parametrize.register_parametrization(layer, "weight", self.LoRA_convertor(layer, rank=rank, lora_alpha=lora_alpha))

    def Enable_lora(self, model, enable=True):
        """
        Enables or disables LoRA for all parametrized layers in the model.

        Args:
            model (nn.Module): The neural network model.
            enable (bool): Flag to enable or disable LoRA.
        """
        for layer in [model.layer1, model.layer2, model.layer3]:
            lora = layer.parametrizations['weight'][0]
            lora.LoRA = enable
            lora.A.requires_grad = enable
            lora.B.requires_grad = enable

    def Disable_lora(self, model, enable=False):
        """
        Disables or enables LoRA for all parametrized layers in the model.

        Args:
            model (nn.Module): The neural network model.
            enable (bool): Flag to disable or enable LoRA.
        """
        for layer in [model.layer1, model.layer2, model.layer3]:
            lora = layer.parametrizations['weight'][0]
            lora.LoRA = not enable
            lora.A.requires_grad = not enable
            lora.B.requires_grad = not enable

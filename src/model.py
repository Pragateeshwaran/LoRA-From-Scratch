import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from lora import LoRA_convertor

class SimpleNN(nn.Module):
    """
    A simple fully connected neural network with three linear layers.
    """
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 10000)
        self.layer2 = nn.Linear(10000, 10000)
        self.layer3 = nn.Linear(10000, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)  # Output logits (no softmax)
        return x

def apply_lora(model, rank=4, lora_alpha=2):
    """
    Applies LoRA parametrization to all linear layers in the model.

    Args:
        model (nn.Module): The neural network model.
        rank (int): Rank of the LoRA matrices.
        lora_alpha (float): Scaling factor for LoRA.
    """
    for layer in [model.layer1, model.layer2, model.layer3]:
        parametrize.register_parametrization(layer, "weight", LoRA_convertor(layer, rank=rank, lora_alpha=lora_alpha))

def Enable_lora(model, enable=True):
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

def Disable_lora(model, enable=False):
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

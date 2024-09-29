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


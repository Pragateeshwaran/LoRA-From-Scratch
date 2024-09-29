# LoRA in Nutshell

import torch
import torch.nn as nn

# Original big neural network
class BigNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(100, 100)  # A big layer with 10,000 parameters

    def forward(self, x):
        return self.layer(x)

# LoRA addon
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = 0.1
        
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scale

# Combine original network with LoRA
class NetworkWithLoRA(nn.Module):
    def __init__(self, original_network):
        super().__init__()
        self.original_network = original_network
        self.lora = LoRALayer(100, 100)
        
    def forward(self, x):
        return self.original_network(x) + self.lora(x)

# Create and use the network
original_net = BigNetwork()
net_with_lora = NetworkWithLoRA(original_net)

# Example input
x = torch.randn(1, 100)

# Use the network
output = net_with_lora(x)

# Disable LoRA
output_without_lora = original_net(x)
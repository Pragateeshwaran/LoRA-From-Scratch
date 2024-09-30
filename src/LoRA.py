import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

class LoRA_scratch(nn.Module):
    def __init__(self, layer, rank, device, alpha=1):
        super().__init__()
        feature_in, feature_out = layer.weight.shape
        self.device = device  
        self.A = nn.Parameter(torch.zeros(feature_in, rank, device=self.device))
        self.B = nn.Parameter(torch.zeros(rank, feature_out, device=self.device))

        self.LoRA = True
        self.scale = torch.tensor(alpha / rank, device=self.device)  # Ensure scale is also on the correct device

    def forward(self, weight):
        if self.LoRA:
            # Ensure that the operation is performed on the same device as the weight tensor
            return weight + (self.A @ self.B).to(self.device) * self.scale.to(self.device)
        return weight

def linear_layer_parameterization(layer, rank, device):
    return LoRA_scratch(layer, rank, device, alpha=1)

def apply_LoRA(model, rank, device, enable=True):
    total_parameters_original = sum(p.numel() for p in model.parameters())
    print(f'Original number of parameters: {total_parameters_original:,}')

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            parametrize.register_parametrization(
                layer, "weight", linear_layer_parameterization(layer, rank, device)
            )

    if enable:
        for name, param in model.named_parameters():
            if 'parametrizations' not in name:
                param.requires_grad = False

    total_parameters_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable LoRA parameters: {total_parameters_lora:,}')

    return model

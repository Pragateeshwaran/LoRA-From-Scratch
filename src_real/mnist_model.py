import torch
import torch.nn as nn
import torch.nn.functional as F

class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 10000)
        self.layer2 = nn.Linear(10000, 20000)
        self.layer3 = nn.Linear(20000, 20000)
        self.layer4 = nn.Linear(20000, 10)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = X.view(-1, 28 * 28)
        X = self.relu(self.layer1(X))
        X = self.relu(self.layer2(X))
        X = self.relu(self.layer3(X))
        X = self.layer4(X) 
        return X


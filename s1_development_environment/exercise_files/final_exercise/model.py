# Filename: model.py
from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super(MyAwesomeModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

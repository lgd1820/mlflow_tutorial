import torch.nn as nn

class Mnist(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(784, 10, bias=True)

    def forward(self, x):
        return self.linear(x)
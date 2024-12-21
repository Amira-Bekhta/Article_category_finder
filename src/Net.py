import torch.nn as nn
import torch.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10000, 5000)
        self.layer2 = nn.Linear(5000, 700)
        self.layer3 = nn.Linear(700, 128)
        self.layer4 = nn.Linear(128, 1)

    def forward(self, input):
        out = self.layer1(input)
        out = F.relu(out)
        out = self.layer2(out)
        out = F.relu(out)
        out = self.layer3(out)
        out = F.relu(out)
        out = self.layer4(out)
        return out



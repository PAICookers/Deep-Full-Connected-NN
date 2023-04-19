from torchsummary import summary
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFCNN(nn.Module):
    def __init__(self) -> None:
        self.fc1 = nn.Linear(1024, 2048)

        self.fc2 = nn.Linear(2048, 4096)

        self.fc3 = nn.Linear(4096, 8192)

        self.fc4 = nn.Linear(8192, 4096)

        self.fc5 = nn.Linear(4096, 2048)

        self.fc6 = nn.Linear(2048, 1024)

        self.fc7 = nn.Linear(1024, 10)

    def forward(self, x) -> Tensor:
        x = x.view(-1, 3, 32, 32)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(F.dropout(x, p=0.2))

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    model = DFCNN()

    print(summary(model, (1, 1, 32, 32)))

    x = torch.rand((1, 1, 32, 32))
    x = x.view(-1, 1, 32, 32)
    x = x.view(x.size(0), -1)
    print(x.size())

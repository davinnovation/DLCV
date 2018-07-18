import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class AlexNetOriginal(nn.Module):
    def __init__(self, n_out=10):
        super(AlexNetOriginal,self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride =4),
                    nn.ReLU(),
                    nn.LocalResponseNorm(size=5, alpha=1e-5, beta=0.75, k=2)
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=96, out_channels = 256, kernel_size = 3, stride = 1, padding=1),
                    nn.ReLU(),
                    nn.LocalResponseNorm(size=5, alpha=1e-5, beta=0.75, k=2)
        )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
        )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
        )
        self.conv5 = nn.Sequential(
                    nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.LocalResponseNorm(size=5, alpha=1e-5, beta=0.75, k=2)
        )
        self.fc1 = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(in_features = 256*6*6, out_features = 4096, bias = True),
                    nn.ReLU(),
                    )
        self.fc2 = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(in_features = 4096, out_features = 4096, bias = True),
                    nn.ReLU()
                    )
        self.fc3 = nn.Linear(in_features = 4096, out_features = n_out)
        

    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1,256*6*6)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out


class AlexNetBn(nn.Module):
    def __init__(self, n_out=10):
        super(AlexNetBn,self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride =4, bias = False),
                    nn.BatchNorm2d(96),
                    nn.ReLU(),
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=96, out_channels = 256, kernel_size = 3, stride = 1, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(384),
                    nn.ReLU()
        )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(384),
                    nn.ReLU()
        )
        self.conv5 = nn.Sequential(
                    nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
                    nn.Linear(in_features = 256*6*6, out_features = 4096, bias = True),
                    nn.ReLU(),
                    )
        self.fc2 = nn.Sequential(
                   nn.Linear(in_features = 4096, out_features = 4096, bias = True),
                   nn.ReLU()
                    )
        self.fc3 = nn.Linear(in_features = 4096, out_features = 10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1,256*6*6)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out

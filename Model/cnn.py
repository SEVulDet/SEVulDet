import torch
from torch import nn
from torch.autograd import Variable


class CnnNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, embed_dim, sequence_length):
        super(CnnNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            kernel_size=(self.kernel_size, self.embed_dim)),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d((1, sequence_length - kernel_size + 1))
        self.dense = torch.nn.Sequential(
            nn.Linear(self.out_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x.view(-1, self.out_channels, self.sequence_length - self.kernel_size + 1))
        x = self.dense(x.view(-1, self.out_channels))
        return x

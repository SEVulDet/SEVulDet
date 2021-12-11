import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_token import *
from cbam import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class CnnNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_width, embed_dim, dropout_rate=0.2):
        super(CnnNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.embed_dim = embed_dim
        self.attention_size = 128
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=int(self.out_channels / 2),
                               kernel_size=9,
                               padding=4)
        self.bn1 = nn.BatchNorm2d(int(self.out_channels / 2))
        self.relu = nn.ReLU()
        self.AttC = CBAM(n_channels_in=int(self.out_channels / 2), reduction_ratio=2, kernel_size=3)
        self.AttT = AttentionToken(self.embed_dim)
        self.bn2 = nn.BatchNorm2d(int(self.out_channels / 2))
        self.conv2 = torch.nn.Conv2d(in_channels=int(self.out_channels / 2),
                                     out_channels=self.out_channels,
                                     kernel_size=(self.kernel_width, self.embed_dim))
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.dp = nn.Dropout(p=dropout_rate)
        self.dense1 = torch.nn.Sequential(
            nn.Linear(self.out_channels * 7, 16),
            nn.ReLU()
        )
        self.dense_output = None
        self.dense2 = nn.Linear(16, 1)

    def forward(self, x):
        # print(x.permute(1, 0, 2))
        x = self.AttT(x.permute(1, 0, 2))
        # x = x.reshape(1, 1, -1, 30)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.AttC(x))
        x = self.relu(self.bn3(self.conv2(x)))
        x = self.dp(x)
        spp = self.spatial_pyramid_pool(x, [int(x.size(2)), int(x.size(3))])
        self.dense_output = self.dense1(spp.view(1, -1))
        return self.dense2(self.dense_output)

    def spatial_pyramid_pool(self, previous_conv, previous_conv_size, out_pool_size=[4, 2, 1]):
        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            h_pad = h_wid * out_pool_size[i] - previous_conv_size[0]
            new_previous_conv = torch.nn.functional.pad(previous_conv, (0, 0, h_pad, 0))
            maxpool = nn.MaxPool2d((h_wid, 1), stride=(h_wid, 1))
            x = maxpool(new_previous_conv)
            if (i == 0):
                spp = x.view(1, self.out_channels, -1)
            else:
                spp = torch.cat((spp, x.view(1, self.out_channels, -1)), 2)
        return spp

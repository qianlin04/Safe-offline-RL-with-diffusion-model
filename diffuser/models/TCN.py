import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import einops
from einops.layers.torch import Rearrange
from .helpers import (
    SinusoidalPosEmb,
)


class TemporalVauleNet(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 2, 1),
        out_dim=1, 
        attention = False,
        kernel_size=2, 
        dropout=0, 
    ):
        super().__init__()
        self.tcn = TemporalConvNet(horizon, transition_dim, dim, dim_mults=dim_mults, kernel_size=kernel_size, dropout=dropout, output_dim=dim)
        self.linear = nn.Linear(dim*horizon, out_dim)

    def forward(self, x, cond, time):
        output = self.tcn(x, cond, time)
        output = output.reshape(output.shape[0], -1)
        pred = self.linear(output)
        return pred


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, embed_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, n_outputs),
            Rearrange('batch t -> batch t 1'),
        )

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x, t):
        out = self.net1(x) + self.time_mlp(t)
        out = self.net2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self, 
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 2, 1),
        output_dim = None,
        attention = False,
        kernel_size=2,
        dropout=0
    ):

        super().__init__()
        self.layers = nn.ModuleList([])

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        
        if output_dim is None:
            output_dim = transition_dim
        num_channels = [*map(lambda m: dim * m, dim_mults)]
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = transition_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, embed_dim=dim, dropout=dropout))
        self.final_conv = nn.Conv1d(num_channels[-1], output_dim, 1)
        
        #self.network = nn.Sequential(*layers)

    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)
        for layer in self.layers:
            x = layer(x, t)
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        # if x.shape[-1]==3 or x.shape[-1]==1:
        #     print("TCN:    ", x[0,0])
        return x


# if __name__ == '__main__':
#     x = torch.rand((64, 32, 15))
#     net = TemporalConvNet(32, 15, 0, output_dim=11)
#     y = net(x)
#     print(y.shape)
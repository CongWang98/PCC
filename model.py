# -*- coding: utf-8 -*-
# @Time    : 2021/01/24
# @Author  : Cong Wang
# @Github ：https://github.com/CongWang98
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(in_c, out_c):
    layers = [
        nn.Linear(in_c, out_c),
        nn.ReLU(True)
    ]
    return layers


class AEncoder(nn.Module):
    def __init__(self, input_dim, inter_dims=[500, 500, 2000], hid_dim=10):
        super().__init__()
        layerlist = block(input_dim, inter_dims[0])
        for i in range(len(inter_dims) - 1):
            layerlist += block(inter_dims[i], inter_dims[i + 1])
        self.encoder = nn.Sequential(*layerlist)
        self.mu = nn.Linear(inter_dims[-1], hid_dim)

    def forward(self, x):
        e = self.encoder(x)
        mu = self.mu(e)
        return mu


class ADecoder(nn.Module):
    def __init__(self, input_dim, inter_dims=[500, 500, 2000], hid_dim=10):
        super().__init__()
        layerlist = block(hid_dim, inter_dims[-1])
        for i in range(len(inter_dims) - 1):
            layerlist += block(inter_dims[- i - 1], inter_dims[- i - 2])
        layerlist.append(nn.Linear(inter_dims[0], input_dim))
        self.decoder = nn.Sequential(*layerlist)

    def forward(self, z):
        x_pred = self.decoder(z)
        return x_pred


class FCAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = AEncoder(
            args.input_dim,
            args.inter_dims,
            args.hid_dim
        )
        self.decoder = ADecoder(
            args.input_dim,
            args.inter_dims,
            args.hid_dim
        )
        self.args = args

    def forward(self, x):
        mu = self.encoder(x)
        self.z_mean = mu
        return self.decoder(mu)


class AEparameter:
    def __init__(self, input_dim, inter_dims, hid_dim):
        self.input_dim = input_dim
        self.inter_dims = inter_dims
        self.hid_dim = hid_dim

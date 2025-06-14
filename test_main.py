#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:24:58 2023

@author: aditib
"""

import argparse
import random
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from model import *
from utils import *
import dgl

device = torch.device("cpu")

adj_mx = pickle.load(open('adj_mat.pickle', 'rb'))
sp_mx = sp.coo_matrix(adj_mx[:, :, 1])
G = dgl.from_scipy(sp_mx)

n_feat = pickle.load(open('tracks.pickle', 'rb'))
num_nodes, _, num_samples = n_feat.shape
df = []
for i in range(num_samples):
    df.append(n_feat[:, :, i])
df = np.array(df)

n_route = num_nodes
lr = 0.001
n_pred = 2
n_his = 2
blocks = [2, 2, 2, 2, 2, 2]
drop_prob = 0
batch_size = 8
epochs = 20
num_layers = 9
control_str = "TNTSTNTST"

W = adj_mx[:, :, 1]
len_val = round(num_samples * 0.1)
len_train = round(num_samples * 0.7)
train = df[:len_train]
val = df[len_train: len_train + len_val]
test = df[len_train + len_val:]

'''
scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)
'''

x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)

loss = nn.MSELoss()
G = G.to(device)
model = STGCN_WAVE(
    blocks, n_his, n_route, G, drop_prob, num_layers, device, control_str
).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf

# training the model
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print(
        "epoch",
        epoch,
        ", train loss:",
        l_sum / n,
        ", validation loss:",
        val_loss,
    )


'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import ChebConv


class TemporalConvLayer(nn.Module):
    """Temporal convolution layer.
    arguments
    ---------
    c_in : int
        The number of input channels (features)
    c_out : int
        The number of output channels (features)
    dia : int
        The dilation size
    """

    def __init__(self, c_in, c_out, dia=1):
        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size = (2, 1), stride = 1, dilation=dia, padding=(0, 1)
        )

    def forward(self, x):
        return torch.relu(self.conv(x))


class SpatioConvLayer(nn.Module):
    def __init__(self, c, Lk):  # c : hidden dimension Lk: graph matrix
        super(SpatioConvLayer, self).__init__()
        self.g = Lk
        self.gc = GraphConv(c, c, activation=F.relu)
        # self.gc = ChebConv(c, c, 3)

    def init(self):
        stdv = 1.0 / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = x.transpose(0, 3)
        x = x.transpose(1, 3)
        output = self.gc(self.g, x)
        output = output.transpose(1, 3)
        output = output.transpose(0, 3)
        return torch.relu(output)


class FullyConvLayer(nn.Module):
    def __init__(self, c):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T,1), 1, dilation=0, padding=(0, 0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, 1, 1, dilation=0, padding=0)
        self.fc = FullyConvLayer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)
'''



'''

class STGCN_WAVE(nn.Module):
    def __init__(
        self, c, T, n, Lk, p, num_layers, device, control_str="TNTSTNTST"
    ):
        super(STGCN_WAVE, self).__init__()
        self.control_str = control_str  # model structure controller
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        cnt = 0
        diapower = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == "T":  # Temporal Layer
                self.layers.append(
                    TemporalConvLayer(c[cnt], c[cnt + 1], dia=2**diapower)
                )
                diapower += 1
                cnt += 1
            if i_layer == "S":  # Spatio Layer
                self.layers.append(SpatioConvLayer(c[cnt], Lk))
            if i_layer == "N":  # Norm Layer
                self.layers.append(nn.LayerNorm([n, c[cnt]]))
        self.output = OutputLayer(c[cnt], T + 1, n)
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x):
        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == "N":
                x = self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.output(x)
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aditib
"""


import random
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from model import *

from sklearn.preprocessing import StandardScaler
from utils import *


import dgl

device = torch.device("cpu")

adj_mx = pickle.load(open('adj_mat.pickle', 'rb'))
sp_mx = sp.coo_matrix(adj_mx[:,:,1])
G = dgl.from_scipy(sp_mx)

n_feat = pickle.load(open('tracks.pickle', 'rb'))
num_nodes, _, num_samples = n_feat.shape
df = []
for i in range(num_samples):
    df.append(n_feat[:,:,i])
df = np.array(df)

n_route = num_nodes
lr = 0.001
n_pred = 2
n_his = 2
blocks = [2,14,28,28,56,112]
drop_prob = 0
batch_size = 8
epochs = 20
num_layers = 9
control_str = "TS" #

W = adj_mx[:,:,1]
len_val = round(num_samples * 0.1)
len_train = round(num_samples * 0.7)
train = df[:len_train]
val = df[len_train : len_train + len_val]
test = df[len_train + len_val :]



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

#training the model
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        #print(x)
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

best_model = STGCN_WAVE(
    blocks, n_his, n_route, G, drop_prob, num_layers, device, args.control_str
).to(device)
best_model.load_state_dict(torch.load(save_path))


l = evaluate_model(best_model, loss, test_iter)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)


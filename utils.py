#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aditib
"""

import numpy as np
import torch


def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(
                model(x).view(len(x), -1).cpu().numpy()
            ).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d**2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE
    


  
def data_transform(data, n_his, n_pred, device):
    # produce data slices for training and testing
    n_route = data.shape[1]
    l = len(data)
    num = l - n_his - n_pred
    #print( num, 2, n_his, n_route)
    x = np.zeros([num, 2, n_his, n_route])
    y = np.zeros([num, n_route,2])
    
    cnt = 0
    for i in range(l - n_his - n_pred):
        head = i
        tail = i + n_his
        x[cnt, :, :, :] = data[head:tail].reshape(2, n_his, n_route)
        y[cnt] = data[tail + n_pred - 1, 2]
        cnt += 1
        #print(i, num, 2, n_his, n_route)
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


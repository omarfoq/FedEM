import torch
import torch.nn.functional as F


def mse(y_pred, y):
    return F.mse_loss(y_pred, y)


def binary_accuracy(y_pred, y):
    y_pred = torch.round(torch.sigmoid(y_pred))  # round predictions to the closest integer
    correct = (y_pred == y).float()
    acc = correct.sum()
    return acc


def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).float()
    acc = correct.sum()
    return acc

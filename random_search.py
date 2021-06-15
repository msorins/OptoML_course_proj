import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random

def estimate_gradient(w, x, y, m, loss):
    
    u = np.random.normal(size = w.shape)
    
    pred_w = sigmoid(x.dot(w))
    pred_u = sigmoid(x.dot(w+m*u))
    
    loss_w = loss(y, pred_w)
    loss_u = loss(y, pred_u)
    
    grad_est = (loss_u - loss_w) * u / m
    return grad_est, loss_w 


def mse(y, x, w):
    """
    Compute the mse loss.
    """
    e = y - sigmoid(x.dot(w))
    mse = e.dot(e) / (2* len(e))
    return mse

def neg_loglikelihood(y, x, w):
    
    pred = sigmoid(x.dot(w))
    return (1-y).dot(np.log(1 - pred)) + y.dot(np.log(pred))

def random_search(epochs, X_train, y_train, X_valid, y_valid, gamma, m, loss):
    
    w = np.zeros(X_train.shape[1])
    losses = []
    valid_losses = []
    for e in range(epochs):
        #pred_train = sigmoid(X_train.dot(w))
        pred_valid = sigmoid(X_valid.dot(w))
        grad, loss_w = estimate_gradient(w, X_train, y_train, m, loss)
        valid_loss = loss(y_valid, pred_valid)
        w = w - gamma * grad
        losses.append(loss_w)
        valid_losses.append(valid_loss)
        if e%10==0:
            print('loss_w = {}'.format(loss_w))
            
    return w, losses, valid_losses

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))
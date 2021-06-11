import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def estimate_gradient(w, x, y, m):
    
    u = np.random.normal(size = w.shape)
    
    loss_w = mse(y, x, w)
    loss_u = mse(y, x, w+m*u)
    
    grad_est = (loss_u - loss_w) * u / m
    return grad_est, loss_w 


def mse(y, x, w):
    """
    Compute the mse loss.
    """
    e = y - x.dot(w)
    mse = e.dot(e) / (2* len(e))
    return mse


def random_search(epochs, X_train, y_train, X_valid, y_valid, gamma, m):
    
    w = np.zeros(X_train.shape[1])
    losses = []
    valid_losses = []
    for e in range(epochs):
        grad, loss_w = estimate_gradient(w, X_train, y_train, m)
        valid_loss = mse(y_valid, X_valid, w)
        w = w - gamma * grad
        losses.append(loss_w)
        valid_losses.append(valid_loss)
        if e%1000==0:
            print('loss_w = {}'.format(loss_w))
            
    return w, losses, valid_losses
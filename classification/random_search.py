import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random

def estimate_gradient(w, x, y, m, loss):
    """
    Estimate gradient of loss function for x with parameter w
    (Gradient oracle)
    """
    
    u = np.random.normal(size = w.shape)
    
    pred_w = sigmoid(x.dot(w))
    pred_u = sigmoid(x.dot(w+m*u))
    
    loss_w = loss(y, pred_w)
    loss_u = loss(y, pred_u)
    grad_est = (loss_u - loss_w) * u / m
    return grad_est, loss_w 

def random_search(epochs, X_train, y_train, X_valid, y_valid, gamma, m, loss, verbose=True):
    """
    Run random search using gradient oracle 
    """
    
    w = np.zeros(X_train.shape[1])
    losses = []
    valid_losses = []
    
    for e in range(epochs):
        
        pred_valid = sigmoid(X_valid.dot(w))
        valid_loss = loss(y_valid, pred_valid)
        
        grad, loss_w = estimate_gradient(w, X_train, y_train, m, loss)
        w = w - gamma * grad
            
        losses.append(loss_w)
        valid_losses.append(valid_loss)
        
        if (e%10==0 and verbose):
            print('loss_w = {}'.format(loss_w))
            print('valid_loss = {}'.format(valid_loss))
            
    return w, losses, valid_losses

def sigmoid(t):
    """
    Simoid activation function
    """
    return 1.0 / (1 + np.exp(-t))
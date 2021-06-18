#!/usr/bin/env python
# coding: utf-8


from random_search import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns#
import math
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier



# ## Load and prepare data
print('Loading data')

df = pd.read_csv('data/framingham.csv')
df = df.dropna()


# ### Train-test split

df_train = df.sample(frac=0.66)
df_valid = df.drop(df_train.index)


# In[5]:


X_train = df_train[['male', 'age', 'education', 'currentSmoker',
                    'cigsPerDay', 'BPMeds', 'prevalentStroke',
                    'prevalentHyp', 'diabetes', 'totChol', 
                    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
                   ]].to_numpy()
y_train = df_train['TenYearCHD'].to_numpy()

X_valid = df_valid[['male', 'age', 'education', 'currentSmoker',
                    'cigsPerDay', 'BPMeds', 'prevalentStroke',
                    'prevalentHyp', 'diabetes', 'totChol', 
                    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
                   ]].to_numpy()
y_valid = df_valid['TenYearCHD'].to_numpy()


# In[6]:


X_train = np.c_[np.ones((y_train.shape[0], 1)), X_train]
X_valid = np.c_[np.ones((y_valid.shape[0], 1)), X_valid]


# In[7]:


X_train.shape


# ### Parameter m - tuning

# In[9]:


m_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
epochs = 20
seed = 321


print('tuning m:')

loss_list = []
valid_loss_list = []

for m in m_vals:
    # set seed to get consistent results
    np.random.seed(seed)
    print('m = {}'.format(m))
    w, losses, valid_losses = random_search(epochs, X_train, y_train, X_valid, y_valid, 1e-5, m, log_loss)
    loss_list.append(losses)
    valid_loss_list.append(valid_losses)
    print('-------------------------')


fig, axs = plt.subplots(2,3,sharey=False, sharex=True, figsize=(18,10))
fig.suptitle('Tuning m')
for i in range(len(axs)):
    for j in range(len(axs[0])):
        n = i+j+2*i
        axs[i,j].plot(list(range(epochs)), loss_list[n], label = 'train')
        axs[i,j].plot(list(range(epochs)), valid_loss_list[n], label = 'test')
        axs[i,j].set_xlabel('Epoch')
        axs[i,j].set_ylabel('Error')
        axs[i,j].set_title('m = {}'.format(m_vals[n]))
        axs[i,j].grid()
        axs[i,j].legend()
        
plt.savefig('figs/m_tuning')

print('tuning completed')
print('final test errors:')
for i in range(len(m_vals)):
    print('m = {}'.format(m_vals[i]))
    print('test error = {}'.format(valid_loss_list[i][-1]))


# determine best m value:

final_losses = [valid_loss_list[i][-1] for i in range(len(m_vals))]
best_error = np.min(final_losses)
best_error_ind = np.where(final_losses == best_error)
best_m = m_vals[best_error_ind[0][0]]

print('best m = {}'.format(best_m))


# #### Plots for m
print('generating training plots for different values of m')

m_vals = [0.1, best_m, 1e-20]

loss_list = []
valid_loss_list = []

for m in m_vals:
    np.random.seed(seed)
    w, losses, valid_losses = random_search(epochs, X_train, y_train, X_valid, y_valid, 1e-5, m, log_loss, verbose=False)
    loss_list.append(losses)
    valid_loss_list.append(valid_losses)


fig, axs = plt.subplots(1,3,sharey=False, sharex=True, figsize=(15,4))
fig.suptitle('Training with different values for m')
for i in range(len(axs)):
    axs[i].plot(list(range(epochs)), loss_list[i], label = 'train')
    axs[i].plot(list(range(epochs)), valid_loss_list[i], label = 'test')
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel('Error')
    axs[i].set_title('m = {}'.format(m_vals[i]))
    axs[i].grid()
    axs[i].legend()
    
plt.savefig('figs/m_examples')


# ### Train with best m value

print('training model with best m value')

np.random.seed(seed)
w1, losses1, valid_losses1 = random_search(epochs, X_train, y_train, X_valid, y_valid, gamma=0.00001, m=best_m, loss=log_loss)


# In[43]:

plt.figure(figsize=(5,5))
plt.plot(list(range(20)), losses1, label = 'train')
plt.plot(list(range(20)), valid_losses1, label = 'test')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid()
plt.legend()
fig.suptitle('Training with best m value')
plt.savefig('figs/zo_classification')


# ## SGD

print('training model with SGD')
model = SGDClassifier(loss = 'log', verbose=0, fit_intercept = True, learning_rate = 'constant', eta0 = 1e-5)#, max_iter=1000)

model.fit(X_train, y_train)

print('training completed')

w = model.coef_

pred_tr = sigmoid(X_train.dot(w.T))
sgd_loss_tr = log_loss(y_train, pred_tr)
print('training loss with SGD optimizer: {}'.format(sgd_loss_tr))
print('training loss with zero-order optimizer: {}'.format(losses1[-1]))

pred = sigmoid(X_valid.dot(w.T))
sgd_loss_valid = log_loss(y_valid, pred)
print('test loss with SGD optimizer: {}'.format(sgd_loss_valid))
print('test loss with zero-order optimizer: {}'.format(valid_losses1[-1]))






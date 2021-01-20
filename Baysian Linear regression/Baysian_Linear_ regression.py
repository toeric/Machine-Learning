#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats


# In[2]:


def gaussian_data_generator(m, s):
    u1 = np.random.uniform(0,1)
    u2 = np.random.uniform(0,1)
    mag = math.sqrt(s) * math.sqrt(-2*np.log(u1))
    z0 = mag * math.cos(math.pi*2*u2) + m
    z1 = mag * math.sin(math.pi*2*u2) + m
    return z0


# In[3]:


def linear_generator(n, a, W):
    x = np.random.uniform(-1,1)
    y  = 0.0
    e = gaussian_data_generator(0, a)
    for i in range(n):
        y += W[i]*(x**i)
    
    y += e
    return (x, y)


# In[4]:


class BayesLinReg:

    def __init__(self, n_features, alpha, beta):
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.mean = np.zeros(n_features)
        self.cov_inv = np.identity(n_features) / alpha

    def learn(self, x, y):
        
        cov_inv = self.cov_inv + self.beta * np.outer(x, x)
        cov = np.linalg.inv(cov_inv)
        mean = cov @ (self.cov_inv @ self.mean + self.beta * y * x)

        self.cov_inv = cov_inv
        self.mean = mean

        return self

    def predict(self, x):

        y_pred_mean = x @ self.mean

        w_cov = np.linalg.inv(self.cov_inv)
        y_pred_var = 1 / self.beta + x @ w_cov @ x.T

        return stats.norm(loc=y_pred_mean, scale=y_pred_var ** .5)


# In[5]:


def print_each_datapoint(point_x, point_y, mean, var, pred_mean, pred_var):
    
    print("Add data point (" + str(point_x) + ", " + str(point_y) + "):")
    print(" ")
    print("Postirior mean:")
    for i in range(len(mean)):
        print(mean[i])
    print(" ")
    print("Posterior variance:")
    print(var)
    print(" ")
    print("Predictive distribution ~ N(" + str(pred_mean) + ", " + str(pred_var) + ")")
    print("--------------------------------------------------")
    


# In[6]:


X_ = []
y_ = []
W = [1,2,3,4]
for i in range(2000):
    tmp_X, tmp_y = linear_generator(4, 1, W)
    tmp_all_X = []
    for j in range(len(W)):
        tmp_all_X.append(tmp_X**j)
    
    X_.append(np.array(tmp_all_X))
    y_.append(np.array([tmp_y]))


# In[7]:


def print_gt(W):
    
    x = [(i-50)/25 for i in range(0, 100, 1)]
    y = []
    y_up = []
    y_down = []
    for i in range(len(x)):
        each_y = 0.0
        for j in range(len(W)):
            each_y += W[j] * (x[i]**j)
        y.append(each_y)
        y_up.append(each_y+1)
        y_down.append(each_y-1)

    plt.figure(figsize=(8,6))
    plt.title("Ground truth")
    plt.plot(x, y,color="black")
    plt.plot(x, y_up,color="red")
    plt.plot(x, y_down,color="red")


# In[8]:


def print_pred(mean, var, X_, y_, b, title):
    
    print_line_x = [(i-50)/25 for i in range(0, 100, 1)]
    print_line_y = []
    print_line_y_up = []
    print_line_y_down = []
    for i in range(len(print_line_x)):
        each_y = 0.0
        for j in range(len(mean)):
            each_y += mean[j] * (print_line_x[i]**j)
        print_line_y.append(each_y)
        
        each_x = []
        for j in range(len(mean)):
            each_x.append(print_line_x[i]**j)
        
        each_x = np.array(each_x)
        each_var  = each_x @ var @ each_x.transpose() + 1/b
        print_line_y_up.append(each_y + each_var)
        print_line_y_down.append(each_y - each_var)

    
    point_x = [X_[i][1] for i in range(len(X_))]
    plt.figure(figsize=(8,6))
    plt.title(title)
    plt.plot(point_x, y_, 'bo')
    plt.plot(print_line_x, print_line_y,color="black")
    plt.plot(print_line_x, print_line_y_up,color="red")
    plt.plot(print_line_x, print_line_y_down,color="red")
        


# In[9]:


def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)


# In[10]:


def Sequential_Estimator(m, s):
    new_existingAggregate = (0, 0, 0)
    print("Data point source function: N(" + str(m) + ", " + str(s) + ")")
    print(" ")
    for i in range(10000):
        
        newValue = gaussian_data_generator(m, s)
        new_existingAggregate = update(new_existingAggregate, newValue)
        
        if 1 <= i < 8:    
            print("Add data point: " + str(newValue))
            (count, mean, M2) = new_existingAggregate
            print("Mean = " + str(mean) + "    " + "Variance = " + str(M2/(count-1)))
#             print(mean, M2/count-1)  
        elif i == 8:
            print(" ")
            print("...")
            print(' ')
        elif 9997 <= i:
            print("Add data point: " + str(newValue))
            (count, mean, M2) = new_existingAggregate
            print("Mean = " + str(mean) + "    " + "Variance = " + str(M2/(count-1)))
            
    


# In[11]:


def Baysian_Linear_regression(b, n, a, W):

    X_ = []
    y_ = []
    for i in range(500):
        tmp_X, tmp_y = linear_generator(n, a, W)
        tmp_all_X = []
        for j in range(len(W)):
            tmp_all_X.append(tmp_X**j)

        X_.append(np.array(tmp_all_X))
        y_.append(np.array([tmp_y]))

    model = BayesLinReg(n, alpha=1, beta=b)
    y_pred = np.empty(len(y_))


    for i, (xi, yi) in enumerate(zip(X_, y_)):
        y_pred[i] = model.predict(xi).mean()
        model.learn(xi, yi)

        if i == 0 or i == 1 or i == 2 or i == len(y_)-2 or i ==len(y_)-1:
            var = np.linalg.inv(model.cov_inv)
            each_x = np.array(xi)
            each_var  = each_x @ var @ each_x.transpose() + 1/model.beta
            print_each_datapoint(xi[1], float(yi), model.mean, var, y_pred[i], each_var)
            if i == 2:
                print(" ")
                print("...")
                print(" ")
                print("--------------------------------------------------")

        if i == 9:
            tmp_X = X_[0:9]
            tmp_y = y_[0:9]
            print_pred(model.mean, np.linalg.inv(model.cov_inv), tmp_X, tmp_y, model.beta, "After 10 incomes")
        elif i == 49:
            tmp_X = X_[0:49]
            tmp_y = y_[0:49]
            print_pred(model.mean, np.linalg.inv(model.cov_inv), tmp_X, tmp_y, model.beta, "After 50 incomes")
        elif i == len(y_)-1:
            print_pred(model.mean, np.linalg.inv(model.cov_inv), X_, y_, model.beta, "Predict result")
        elif i == 0:
            print_gt(W)


# In[12]:


# b = 1
# n = 3
# a = 3
# W = [1,2,3]

# b = 100
# n = 4
# a = 1
# W = [1,2,3,4]

b = 1
n = 4
a = 4
W = [1,2,3,4]


# In[13]:


Baysian_Linear_regression(b, n, a, W)


# In[14]:


Sequential_Estimator(3,5)


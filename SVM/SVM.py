#!/usr/bin/env python
# coding: utf-8

# In[49]:


from libsvm.svmutil import *
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


X_test = genfromtxt('ML_HW05-2/X_test.csv', delimiter=',')
X_train = genfromtxt('ML_HW05-2/X_train.csv', delimiter=',')
Y_test = genfromtxt('ML_HW05-2/Y_test.csv', delimiter=',')
Y_train = genfromtxt('ML_HW05-2/Y_train.csv', delimiter=',')


# ## Part1

# In[124]:


def show_confusion_matrix(real, pre):
    
    confusion_matrix = np.zeros((5, 5), dtype=int)
    
    for i in range(len(real)):
            
        confusion_matrix[int(real[i])-1][int(pre[i])-1] += 1
    
    sns.heatmap(confusion_matrix, cmap='Greens', annot=True, fmt='d')
    plt.xlabel("Predict")
    plt.ylabel("Real")  
    x = [1,2,3,4,5]
    xi = [0.5,1.5,2.5,3.5,4.5]
    plt.xticks(xi, x)
    
    y = [1,2,3,4,5]
    yi = [0.5,1.5,2.5,3.5,4.5]
    plt.yticks(yi, y)


# In[125]:


def different_kernel_svm(kernel_type):
    prob  = svm_problem(Y_train, X_train)
    if kernel_type=="linear":
        param = svm_parameter('-t 0')
    elif kernel_type=="polynomial":
        param = svm_parameter('-t 1')
    else:
        param = svm_parameter('-t 2')
        
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)
    
    confusion_matrix = show_confusion_matrix(Y_test, p_label)


# In[126]:


different_kernel_svm("linear")


# In[127]:


different_kernel_svm("polynomial")


# In[128]:


different_kernel_svm("RBF")


# ## Part2

# In[140]:


def C_SVC(cost, gamma):
    prob  = svm_problem(Y_train, X_train)
    param = svm_parameter('-t 2 -c ' + str(cost) + ' -g ' + str(gamma))
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)
    return p_acc[0]


# In[158]:


def grid_search(all_cost, all_gamma):
    
    result_matrix = np.zeros((len(all_cost), len(all_gamma)))
    for i in range(len(all_cost)):
        for j in range(len(all_gamma)):
            result_matrix[i][j] = C_SVC(all_cost[i], all_gamma[j])
    
    return result_matrix


# In[176]:


def draw_result_matrix(all_cost, all_gamma, grid_search_result):
    
    sns.heatmap(grid_search_result, cmap='Blues', annot=True, fmt='.2f')
    
    plt.xlabel("Gamma")
    plt.ylabel("Cost")  
    
    xi = []
    yi = []
    for i in range(len(all_gamma)):
        xi.append(i+0.5)
    plt.xticks(xi, all_gamma)
    
    for i in range(len(all_cost)):
        yi.append(i+0.5)
    plt.yticks(yi, all_cost)
    


# In[159]:


all_cost = [1, 5, 10, 25, 50]
all_gamma = [0.0001, 0.001,0.01,0.1,1]
grid_search_result1 = grid_search(all_cost, all_gamma)


# In[169]:


draw_result_matrix(all_cost, all_gamma, grid_search_result_1)


# In[173]:


all_cost = [5, 7.5, 10]
all_gamma = [0.008, 0.01, 0.02, 0.03, 0.04, 0.05]
grid_search_result2 = grid_search(all_cost, all_gamma)


# In[179]:


draw_result_matrix(all_cost, all_gamma, grid_search_result2)


# ## Part3

# In[262]:


def linear_kernel(a, b):
    a = np.array(a)
    b = np.array(b)
    return a.T @ b

def RBF_kernel(a, b):
    gamma = 1/784
    a_b = np.array([a[i] - b[i] for i in range(len(a))])
    return np.exp(-1*gamma*(a_b.T @ a_b))


# In[263]:


def build_K(data1, data2):
    N1 = len(data1)
    N2 = len(data2)

    K = []
    for i in range(N1):
        tmp = []
        tmp.append(i+1)
        for j in range(N2):          
            tmp.append(linear_kernel(data1[i], data2[j]) + RBF_kernel(data1[i], data2[j]))
        K.append(tmp)
        if i%500 ==0:
            print(i)
        
    return K


# In[ ]:


K_train = build_K(X_train, X_train)
K_test = build_K(X_test, X_train)
model = svm_train(Y_train, K_train, '-t 4')
p_label, p_acc, p_val = svm_predict(Y_test, K_test, model)


# ## Discussion

# In[283]:


def ignore_some_data(n):
    new_train_Y = []
    new_train_X = []
    for i in range(len(Y_train)):
        if i <n*1000-1000 or i > n*1000-1:
            new_train_Y.append(Y_train[i])
            new_train_X.append(X_train[i])
    new_train_Y = np.array(new_train_Y)
    new_train_X = np.array(new_train_X)
    
    prob  = svm_problem(new_train_Y, new_train_X)
    param = svm_parameter('-t 0')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)
    return p_acc[0]


# In[287]:


result_ = []
for i in range(1, 6):
    print("Ignore " + str(i) + ":")
    result_.append(ignore_some_data(i))
    print(" ")


# In[289]:


x = [1,2,3,4,5]
plt.xlabel("Ignore label")
plt.ylabel("Accuracy")  
plt.bar(x, result_)


# In[ ]:





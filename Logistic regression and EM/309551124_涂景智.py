#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:


def gaussian_data_generator(m, s):
    u1 = np.random.uniform(0,1)
    u2 = np.random.uniform(0,1)
    mag = math.sqrt(s) * math.sqrt(-2*np.log(u1))
    z0 = mag * math.cos(math.pi*2*u2) + m
    z1 = mag * math.sin(math.pi*2*u2) + m
    return z0


# In[3]:


def gen_data(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2):
    
    data_1 = []
    data_2 = []
    data_1_label = []
    data_2_label = []
    
    real_point_1 = []
    real_point_2 = []
    for i in range(N):
        x1 = gaussian_data_generator(mx1, vx1)
        y1 = gaussian_data_generator(my1, vy1)
        x2 = gaussian_data_generator(mx2, vx2)
        y2 = gaussian_data_generator(my2, vy2)
        data_1.append([1, x1, y1])
        data_2.append([1, x2 ,y2])
        real_point_1.append([x1, y1])
        real_point_2.append([x2 ,y2])
        data_1_label.append(0)
        data_2_label.append(1)
        
    return real_point_1+real_point_2, data_1+data_2, data_1_label+data_2_label


# In[4]:


def cal_J(w, X, Y):
    tmp = 1 / (1 + np.exp(- (X @ w)))
    return X.T @ (Y - tmp)


# In[5]:


def cal_Hessian(A, w):
    
    D = np.zeros((len(A), len(A)))
    A = np.array(A)
    
    for i in range(len(D)):
        tmp = np.exp(-A[i]@w)
        D[i][i] = tmp / (1+tmp)**2
    
    return A.T @ D @ A


# In[6]:


def gradient_descent(X, Y, lr = 0.001):
    
    X = np.array(X)
    Y = np.array(Y)
    w = np.zeros(3)
    
    converge_num = 1000
    
    while converge_num > 0.01:
        
        J = cal_J(w, X, Y)
        w = w + lr*J
        converge_num = np.linalg.norm(J)
        
    return w


# In[7]:


def Newton_method(X, Y, lr = 0.001):
    
    X = np.array(X)
    Y = np.array(Y)
    w = np.zeros(3)
    count = 0
    converge_num = 0
    next_converge_num = 1000
    while next_converge_num - converge_num > 0.25:
        count += 1
        
        H = cal_Hessian(X, w)
        
        if np.linalg.matrix_rank(H)!=3:
            print("Use gradient descent")
            w = gradient_descent(X, Y, lr = 0.001)
            break
        else:
            inv_H = np.linalg.inv(H)
            w = w + lr * (inv_H @ cal_J(w, X, Y))
            next_converge_num = np.linalg.norm(inv_H)
#     print(next_converge_num - converge_num)
    return w  


# In[8]:


def plot_scatter(class_1, class_2, img_name):
    
    name = img_name

    plt.scatter(class_1[:,0], class_1[:,1], c="blue")
    plt.scatter(class_2[:,0], class_2[:,1], c="red")  
    plt.title(img_name)
    plt.show()
    


# In[9]:


def predict(X, w, real_Y):
    
    y_pred_tmp = 1 / (1 + np.exp(- (X @ w)))
    y_pred = []
    for i in range(len(y_pred_tmp)):
        if y_pred_tmp[i]<0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)
            
    #count acc:
    
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == real_Y[i]:
            count +=1
    
#     print(count/len(y_pred))
    
    return y_pred


# In[14]:


# N = 50
# mx1 = 1
# my1 = 1
# mx2 = 10
# my2 = 10
# vx1 = 2 
# vy1 = 2
# vx2 = 2 
# vy2 = 2 

N = 50
mx1 = 1
my1 = 1
mx2 = 3
my2 = 3
vx1 = 2 
vy1 = 2
vx2 = 4 
vy2 = 4 


# In[11]:


def Logistic_regression(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2):
    
    real_point, data,  label = gen_data(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2)
    real_point_c1 = np.array(real_point[:50])
    real_point_c2 = np.array(real_point[50:])
    plot_scatter(real_point_c1, real_point_c2, "Grount truth")

    
    w_gd = gradient_descent(data, label, lr = 0.001)
    w_nm = Newton_method(data, label, lr = 0.001)
    ans_gd = predict(data, w_gd, label)
    ans_nm = predict(data, w_nm, label)
    
    gd_point_c1 = []
    gd_point_c2 = []
    nm_point_c1 = []
    nm_point_c2 = []
    
    for i in range(len(ans_gd)):
        if ans_gd[i] == 0:
            gd_point_c1.append(real_point[i])
        else:
            gd_point_c2.append(real_point[i])
            
    for i in range(len(ans_nm)):
        if ans_nm[i] == 0:
            nm_point_c1.append(real_point[i])
        else:
            nm_point_c2.append(real_point[i])
    
    plot_scatter(np.array(gd_point_c1), np.array(gd_point_c2), "Gradient descent")
    plot_scatter(np.array(nm_point_c1), np.array(nm_point_c2), "Newton's method")
    
    show_result(w_gd, w_nm, ans_gd, ans_nm, label)
    
#     print(w_gd)
#     print(w_nm)


# In[12]:


def show_result(w_gd, w_nm, ans_gd, ans_nm, label):
    
    N = len(label)
    
    TP_gd = FP_gd = FN_gd = TN_gd = 0 
    TP_nm = FP_nm = FN_nm = TN_nm = 0 
    
    for i in range(N):
        if label[i] == 0 and ans_gd[i] == 0:
            TN_gd += 1
        elif label[i] == 0 and ans_gd[i] == 1:
            FP_gd += 1
        elif label[i] == 1 and ans_gd[i] == 0:
            FN_gd += 1
        else:
            TP_gd += 1
            
    for i in range(N):
        if label[i] == 0 and ans_nm[i] == 0:
            TN_nm += 1
        elif label[i] == 0 and ans_nm[i] == 1:
            FP_nm += 1
        elif label[i] == 1 and ans_nm[i] == 0:
            FN_nm += 1
        else:
            TP_nm += 1
        
    
    print("Gradient descent:")
    print("")
    print("w")
    for i in range(3):
        print("  " + str(w_gd[i]))
    print(" ")
    print("Confusion Matrix:")
    print("              " + "Predict cluster 1 Predict cluster 2")
    print("Is cluster 1        " + str(TN_gd) + "                " + str(FP_gd))
    print("Is cluster 2        " + str(FN_gd) + "                " + str(TP_gd))
    print("Sensitivity (Successfully predict cluster 1): " + str(TN_gd/(TN_gd+FP_gd)))
    print("Sensitivity (Successfully predict cluster 2): " + str(TP_gd/(FN_gd+TP_gd)))
    
    print(" ")
    print("-----------------------------------")
    
    print("Newton's method:")
    print("")
    print("w")
    for i in range(3):
        print("  " + str(w_nm[i]))
    print(" ")
    print("Confusion Matrix:")
    print("              " + "Predict cluster 1 Predict cluster 2")
    print("Is cluster 1        " + str(TN_nm) + "                " + str(FP_nm))
    print("Is cluster 2        " + str(FN_nm) + "                " + str(TP_nm))
    print("Sensitivity (Successfully predict cluster 1): " + str(TN_nm/(TN_nm+FP_nm)))
    print("Sensitivity (Successfully predict cluster 2): " + str(TP_nm/(FN_gd+TP_nm)))
    
    


# In[15]:


Logistic_regression(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def C_N(up, down):
    
    a = 1.0
    b = 1.0
    c = 1.0
    
    for i in range (up):
        a = a*(i+1)
    for i in range (down):
        b = b*(i+1)
    for i in range (up - down):
        c = c*(i+1)
        
    return a/(b*c)


# In[6]:


def Cal_P(n, y, p):
    return C_N(n, y)*(p**y)*((1-p)**(n-y))


# In[28]:


def Online_learning(input_a, intput_b):
    
    f = open("testfile.txt", 'r')
    idx = 0
    each_line = f.readline()
    a = input_a
    b = intput_b
    Likelihood = 0.0
    while each_line!="":
        idx += 1
        each_line = str(each_line)
        each_line = each_line.strip("\n")
        is_1 = 0
        is_0 = 0
        for i in range(len(each_line)):
            if each_line[i]=="0":
                is_0 += 1
            elif each_line[i]=="1":
                is_1 += 1
        print("case " + str(idx) + ": " + each_line)
        print("Likelihood: " + str(Cal_P(is_1 + is_0, is_1, is_1/(is_1 + is_0))))
        print("Beta prior: a = " + str(a) + " b = " +  str(b))


        a += is_1
        b += is_0

        print("Beta posterior: a = " + str(a) + " b = " +  str(b))
        print("")


        each_line = f.readline()


# In[30]:


Online_learning(10, 1)


# In[ ]:





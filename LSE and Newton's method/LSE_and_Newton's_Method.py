#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import math
import matplotlib.pyplot as plt


# In[64]:


def LUDecompose (table): 
    rows,columns=np.shape(table)
    L=np.zeros((rows,columns))
    U=np.zeros((rows,columns))
    if rows!=columns:
        return
    for i in range (columns):
        for j in range(i-1):
            sum=0
            for k in range (j-1):
                sum+=L[i][k]*U[k][j]
            L[i][j]=(table[i][j]-sum)/U[j][j]
        L[i][i]=1
        for j in range(i-1,columns):
            sum1=0
            for k in range(i-1):
                sum1+=L[i][k]*U[k][j]
            U[i][j]=table[i][j]-sum1
    return L,U


# In[65]:


def transposeMatrix(m):
    return list(map(list,zip(*m)))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors


# In[66]:


def Build_A_b(n):
    f = open("testfile.txt")
    line = f.readline()
    A = []
    b = []
    while line:
        each_row = []
        line = line.replace('\n', '')
        x, y = line.split(",")
        each_row.append(1)
        for i in range(1, n):
            each_row.append(pow(float(x),i))
        A.append(each_row)
        b.append(float(y))
        line = f.readline()
    f.close()
    
    return A, b


# In[67]:


def LSE(A, b, Lambda, n):
    b = np.array(b)
    dim = n
    A_T = transposeMatrix(A)
    A_T = np.array(A_T)
    mul_I = Lambda * np.identity(dim)
    LSE_matrix = (np.matmul(A_T, A)+ mul_I).tolist()
    L, U = LUDecompose(LSE_matrix)
    L = L.tolist()
    U = U.tolist()
    L_inverse = np.array(getMatrixInverse(L))
    U_inverse = np.array(getMatrixInverse(U))
    LSE_matrix_inverse = np.matmul(U_inverse, L_inverse)
    tmp = np.matmul(LSE_matrix_inverse, A_T)
    x = np.matmul(tmp, b)
    return x


# In[68]:


def cal_error(A, b, x):
    total_error = 0.0
    idx = 0
    for each in A:
        sum = 0
        for i in range(len(each)):
            sum += each[i]*x[i]
        
        total_error += pow(sum-b[idx],2)
        idx += 1
        
    return total_error
            


# In[69]:


def Step_Newton_mrthod(A, b, x):
    
    Hession_matrix = 2 * np.matmul(np.array(transposeMatrix(A)), np.array(A))
    Inverse_Hession_matrix = np.array(getMatrixInverse(Hession_matrix.tolist()))
    g_f = 2 * np.matmul(np.array(transposeMatrix(A)) ,(np.matmul(np.array(A), np.array(x)) - np.array(b)))
    new_x = x - np.matmul(Inverse_Hession_matrix, g_f)
    return new_x


# In[70]:


def Newton_method(A, b, n):
    x = []
    for i in range(n):
        x.append(0.0)
        
    for i in range(1):
        x = Step_Newton_mrthod(A, b, x)
 
    return x


# In[71]:


def draw_map(LSE_result):
    f = open("testfile.txt")
    line = f.readline()
    all_x = []
    all_y = []
    while line:
        each_row = []
        line = line.replace('\n', '')
        x, y = line.split(",")
       
        all_x.append(float(x))
        all_y.append(float(y))
        line = f.readline()
    f.close()
    plt.scatter(all_x, all_y)
    
    curve_y = []
    curve_x = np.linspace(-6, 6, 100)
    
    for each in curve_x:
        tmp_y = 0.0
        for i in range(len(LSE_result)):
            tmp_y += each**i * LSE_result[i]
        curve_y.append(tmp_y)
        
        
    
    plt.plot(curve_x, curve_y, color='black')
    plt.xticks(np.arange(-6, 6, 1))
    plt.yticks(np.arange(0, 100, 10))
    plt.xlim(-6, 6)
    plt.ylim(-10, 110)
    plt.show()


# In[72]:


def run(n, Lambda):
    
    A, b = Build_A_b(n)
    LSE_result = LSE(A, b, Lambda, n)
    Newton_result = Newton_method(A, b, n)
    
    print("LSE:")
    print("Fitting line: ", end='')
    for i in range(n-1, -1, -1):
        if i != 0:
            print(LSE_result[i], end='')
            print("X^{}".format(i), end='')
            
            if LSE_result[i-1]>0:
                print(" + ", end='')
            else:
                print(" ", end='')
        else:
            print(LSE_result[i])
    
    print("Total error: ", end='')
    LSE_error = cal_error(A, b, LSE_result)
    print(LSE_error)
    
    print("")
    print("Newton's Method:")
    print("Fitting line: ", end='')
    for i in range(n-1, -1, -1):
        if i != 0:
            print(Newton_result[i], end='')
            print("X^{}".format(i), end='')
            
            if Newton_result[i-1]>0:
                print(" + ", end='')
            else:
                print(" ", end='')
        else:
            print(Newton_result[i])
            
    print("Total error: ", end='')
    Newton_error = cal_error(A, b, Newton_result)
    print(Newton_error)
    
    draw_map(LSE_result)
    draw_map(Newton_result)
            
            


# In[76]:


run(3, 10000)


# In[ ]:





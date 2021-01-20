#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import imageio


# In[2]:


img = cv2.imread('ML_HW06/image1.png')
all_point = []
for i in range(100):
    for j in range(100):
        each_point = np.append(img[i][j], [i, j])
        all_point.append(each_point)
all_point = np.array(all_point)


# In[3]:


def cal_gram(data):
    

    tmpC = pdist(data[:, :3])
    tmpC = squareform(tmpC)

    tmpS = pdist(data[:, 3:])
    tmpS = squareform(tmpS)

    cal_gram = np.exp(-1/100000*tmpC**2) * np.exp(-1/100000*tmpS**2)

    return cal_gram


# In[4]:


def draw_img(all_cluster, idx):
    cluster = all_cluster[idx]
    draw_matrix = np.zeros((100,100))
    for i in range(10000):
        if cluster[1][i] == 1:
            draw_matrix[int(i/100)][int(i%100)] = 1
            
    plt.imshow(draw_matrix, cmap="Greys")
    plt.savefig('GIF/it' + str(idx+1) + '.png')


# In[5]:


global_U = np.zeros((10000,2))


# In[6]:


def spec_pre(n_cut):
    
    global global_U 
    
    K = 2
    
    old_gram = cal_gram(all_point)
    W = np.copy(old_gram)
    tmp_d = np.sum(W, axis=0)
    D = np.zeros((10000, 10000))
    np.fill_diagonal(D, tmp_d)
    L = np.array([])
    if n_cut == 0:
        L = np.subtract(D, W)
    elif n_cut == 1:
        sqrtD = np.sqrt(D)
        L = np.linalg.inv(sqrtD) @ np.subtract(D, W) @ np.linalg.inv(sqrtD)
    e_value, e_vector = np.linalg.eigh(L)
    U = np.copy(e_vector[: , :K])
    

    
    if n_cut == 1:
        norm = np.linalg.norm(U, axis=1)
        for i in range(len(U)):
            U[i] = U[i]/norm[i]
            
    global_U = np.copy(U)
    print(U)
    
    tmp_dis = pdist(U, metric='euclidean')
    new_gram = squareform(tmp_dis)
    return new_gram


# In[7]:


def kernel_kmeans(GRAM, num_iter, plusplus=0,):
    
    K = 2
    
    all_result = []
    cluster = np.zeros((K,10000))
    new_Cluster = np.zeros((K,10000))
    
    if plusplus == 0:
        for i in range(10000):
            if i % 100 > 100/K-1:
                cluster[1][i] = 1
            else:
                cluster[0][i] = 1
    else:
        center = np.zeros(K)
        center[0] = np.random.randint(10000)
        p = np.zeros(10000)
        
        for i in range(len(p)):
            p[i] = GRAM[int(center[0])][i]
        p_sum = np.sum(p)
        
        rand_num = np.random.rand() * p_sum
        
        for i in range(10000):
            if rand_num < p[i]:
                center[1] = i
                break
            rand_num  -= p[i]
        for i in range(10000):
            if GRAM[int(center[0])][i] > GRAM[int(center[1])][i]:
                cluster[0][i] = 1
            else:
                cluster[1][i] = 1
        



    all_result.append(cluster)
    
    pixel_min = np.zeros(10000)
    for i in range(10000):
        pixel_min[i] = np.Inf

    for e in range(num_iter):

        print(e)

        clusterCount = np.zeros(K)
        for i in range(K):
            clusterCount[i] = np.sum(cluster[i])

        print(clusterCount)

        right_part = np.zeros(K)
        for c in range(K):

            row_c = np.array([cluster[c]])
            index_array = row_c.T @ row_c
            for i in range(10000):
                for j in range(10000):
                    if index_array[i][j] == 1:
                        right_part[c] += GRAM[i][j]

            right_part = right_part/(clusterCount[c]**2)


            for pixel_index in range(10000):

                middle_part = 0
                row_GRAM = np.array([GRAM[pixel_index]])
                middle_part = row_c @ row_GRAM.T
                middle_part *= -2 / clusterCount[c] 

                if 1 + middle_part + right_part[c] < pixel_min[pixel_index]:
                    pixel_min[pixel_index] = 1 + middle_part + right_part[c]

                    for i in range(K):
                        new_Cluster[i][pixel_index] = 0

                    new_Cluster[c][pixel_index]=1

        cluster = np.copy(new_Cluster)
        all_result.append(cluster)
        
    return all_result


# In[8]:


def compose_gif(total_it):
    all_path = []
    for i in range(total_it):
        all_path.append("GIF/it" + str(i+1) + ".png") 
    gif_images = []
    for path in all_path:
        gif_images.append(imageio.imread(path))
    imageio.mimsave("test.gif",gif_images,fps=1)


# In[ ]:





# In[9]:


def Clustering(k_mean=0, n_cut = 1, kpp = 0):
    
    GRAM = np.zeros((10000,10000))
    
    if k_mean==0:
        GRAM = cal_gram(all_point)
        num_iter = 5 
    else:
        GRAM = spec_pre(n_cut)
        num_iter = 2
        if kpp == 1:
            num_iter-=1
        
    all_result = kernel_kmeans(GRAM, num_iter, kpp)
    for i in range(num_iter+1):
        draw_img(all_result, i)
    compose_gif(num_iter+1)
    
    return all_result


# In[10]:


all_result = Clustering(1, 1, 0)


# In[11]:


all_result = np.array(all_result)


# In[12]:


cluster_1_x = np.array([])
cluster_1_y = np.array([])
cluster_2_x = np.array([])
cluster_2_y = np.array([])
for i in range(10000):
    if all_result[2][0][i] == 1:
        cluster_1_x = np.append(cluster_1_x, global_U[i][0])
        cluster_1_y = np.append(cluster_1_y, global_U[i][1])
    else:
        cluster_2_x = np.append(cluster_2_x, global_U[i][0])
        cluster_2_y = np.append(cluster_2_y, global_U[i][1])


# In[13]:


plt.scatter(cluster_1_x, cluster_1_y,color='red')
plt.scatter(cluster_2_x, cluster_2_y,color='blue')
plt.show()


# In[ ]:





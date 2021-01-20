#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train_image = "data/train-images-idx3-ubyte"
train_label = "data/train-labels-idx1-ubyte"
test_image = "data/t10k-images-idx3-ubyte"
test_label = "data/t10k-labels-idx1-ubyte"


# In[3]:


def read_image(image_file, mode):
    if mode == 0:
        div_num = 8
    else:
        div_num = 1
    all_images = []
    image = []
    with open(image_file, "rb") as f:
        num = 0
        byte = f.read(4)

        while byte != b"":
            num += 1
            if num < 4:
                byte = f.read(4)
            else:
                byte = f.read(1)
                byte_in_int = int.from_bytes(byte, "big")
                image.append(int(byte_in_int/div_num))
                if len(image) > 783:
                    all_images.append(image)
                    image = []

    return all_images


# In[4]:


def read_label(label_file):

    all_labels = []
    
    with open(label_file, "rb") as f:
        num = 0
        byte = f.read(4)

        while byte != b"":
            num += 1
            if num < 2:
                byte = f.read(4)
            else:
                byte = f.read(1)
                byte_in_int = int.from_bytes(byte, "big")
                if byte != b"":
                    all_labels.append(byte_in_int)

    return all_labels


# In[5]:


def normalize_ans(ans):

    _sum = 0.0
    norm_ans = []
    for i in range(10):
        _sum += ans[i]

    for i in range(10):
        norm_ans.append(ans[i]/_sum)

    return norm_ans


# In[6]:


def show_accuracy(predict_ans, test_class):
    predict_ans = predict_ans.tolist()
    count = 0
    for i in range(len(test_class)):
        if test_class[i] == int(predict_ans[i]):
            count += 1
    print("Error rate:" + str(1 - count/len(test_class)))


# In[7]:


def print_Postirior(total_normalize_ans, predict_ans, test_class):
    for i in range(2):
        print("Postirior (in log scale):")
        for j in range(10):
            print(str(j) + ": " + str(total_normalize_ans[i][j]))
        print("Prediction: " + str(int(predict_ans[i]))+ ", Ans: " + str(test_class[i]))
        print("")


# In[8]:


def Cal_P(mode, Count, Class_Count):
        
    P = np.zeros((28, 28, 32, 10))
    for i in range(28):
        for j in range(28):
            for k in range(32):     
                for w in range(10):
                    P[i][j][k][w] = Count[i][j][k][w] / Class_Count[i][j][w]
                    if P[i][j][k][w]==0.0:
                        P[i][j][k][w] = 0.0001

                    P[i][j][k][w] = np.log10(P[i][j][k][w]) 
        
    return P


# In[17]:


all_image = []
def print_imagination(statis_mom):
    
    print("Imagination of numbers in Bayesian classifier:")
    
    for class_idx in range(10):

        img = np.zeros((28,28))

        print(str(class_idx)+":")

        total_ = []
        total_mean = 0.0
        for i in range(28):
            for j in range(28):
                img[i][j] = statis_mom[i][j][class_idx][0]
                total_.append(img[i][j])

        total_mean = np.mean(np.array(total_))
        for i in range(28):
            for j in range(28):
                if img[i][j]>total_mean:
                    img[i][j] = 1
                else:
                    img[i][j] = 0
    #     plt.imshow(img, cmap="Greys")
        all_image.append(img)
        print(img)


# In[33]:


def Naive_Bayes_classifier(mode):
    
    train_data = read_image(train_image, mode)
    test_data = read_image(test_image, mode)
    train_class = read_label(train_label)
    test_class = read_label(test_label)
    
    label_count = np.zeros((10))

    for each in train_class:
        label_count[each] +=1

    _sum = 0
    for i in range(len(label_count)):
        _sum += label_count[i]

    for i in range(len(label_count)):
        label_count[i] = label_count[i]/_sum

    
    print("Load data sucess!")
    
#     if mode == 1:
#         Gray_level_amount = 256
#     else:
#         Gray_level_amount = 32
        
    statis_usedata = []
    for i in range (28):
        j_list = []
        for j in range (28):
            w_list = []
            for w in range(10):
                w_list.append([])
            j_list.append(w_list)
        statis_usedata.append(j_list)


    for image_idx in range(len(train_data)):
        for i in range(784):
            x = int(i/28)
            y = int(i%28)
            Gray_level = train_data[image_idx][i]
            Class = train_class[image_idx]
            statis_usedata[x][y][Class].append(Gray_level)

    statis_mom = np.zeros((28, 28, 10, 2))
    for i in range(28):
        for j in range(28):
            for w in range(10):
                tmp = np.array(statis_usedata[i][j][w])
                statis_mom[i][j][w][0] = np.mean(tmp)
                statis_mom[i][j][w][1] = np.var(tmp)
                if statis_mom[i][j][w][1] == 0:
                    statis_mom[i][j][w][1] = 10000
                
    print("Calculate mean & variance sucess!")

    predict_ans = np.zeros((len(test_data)))
    total_normalize_ans = []    
    
    if mode == 0:
    
        Count = np.zeros((28, 28, 32, 10))
        Class_Count = np.zeros((28, 28, 10))
        img_idx = 0
        for each_image in train_data:
            for i in range(len(each_image)):
                x = int(i/28)
                y = int(i%28)
                Gray_level = each_image[i]
                Class = train_class[img_idx]
                Count[x][y][Gray_level][Class] += 1
                Class_Count[x][y][Class] += 1

            img_idx += 1

        P = Cal_P(mode, Count, Class_Count)
                
        for image_idx in range(len(test_data)): 
            ans = np.zeros((10))
            for i in range(28):
                for j in range(28):
                    gray_level = test_data[image_idx][28*i+j]
                    for _class in range(10):
                        ans[_class] = ans[_class] + P[i][j][gray_level][_class]

    #         for i in range(len(ans)):
    #             ans[i] = ans[i] * label_count[i]

            max_ans = 0.0
            idx = 0
            for i in range(10):
                if i == 0:
                    max_ans = ans[i]
                if ans[i] > max_ans:
                    idx = i
                    max_ans = ans[i]

            predict_ans[image_idx] = idx
            total_normalize_ans.append(normalize_ans(ans))
        
    else:
        
        predict_ans = np.zeros((len(test_data)))
        very_small_number = 0.0000000000000000001

        for img_idx in range(len(test_data)):
            ans = np.zeros((10))
            for w in range(10):
                ratio = []
                for i in range(28):
                    for j in range(28):
                        coef = 1 / pow(2 * np.pi * statis_mom[i][j][w][1] + very_small_number, 0.5)
                        normal = -0.5 * (pow(test_data[img_idx][28*i+j] - statis_mom[i][j][w][0], 2) / (statis_mom[i][j][w][1]+ very_small_number))
                        ratio.append(np.log(coef) + normal)

                ans[w] = np.sum(ratio)
                
            total_normalize_ans.append(normalize_ans(ans))
            ans = ans.tolist()
            predict_ans[img_idx] = ans.index(max(ans))
            
            if img_idx%1000==0:
                print(str(img_idx)+"th image")
            

    
    print_Postirior(total_normalize_ans, predict_ans, test_class)
    print("")
    print_imagination(statis_mom)
    print("")
    show_accuracy(predict_ans, test_class)


# In[34]:


Naive_Bayes_classifier(0)


# In[16]:


Naive_Bayes_classifier(1)


# In[29]:


plt.imshow(all_image[9], cmap="Greys")


# In[ ]:





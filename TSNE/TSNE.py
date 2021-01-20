#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pylab
import imageio
import matplotlib.pyplot as plt


# In[6]:


dir_path = "./ML_HW07/tsne_python/"
X = np.loadtxt(dir_path + "mnist2500_X.txt")
labels = np.loadtxt(dir_path + "mnist2500_labels.txt")


# In[7]:


def draw_img(Y, labels, idx):
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.savefig('GIF/it' + str(idx) + '.png')


# In[8]:


def compose_gif():
    path_idx = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    all_path = []
    for i in path_idx:
        all_path.append("GIF/it" + str(i) + ".png") 
    gif_images = []
    for path in all_path:
        gif_images.append(imageio.imread(path))
    imageio.mimsave("test.gif",gif_images,fps=1)


# In[43]:


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, istsne = True):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    
    all_C = []

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4 # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
            
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        if istsne :
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            num = np.exp(-np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            if istsne :
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            else:
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) +                 (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        
        if (iter+1)%100 == 0: 
            draw_img(Y, labels, iter+1)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            all_C.append(C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y, P, Q, all_C


# In[44]:


print("Run tsne")
Y, P, Q, all_C = tsne(X, 2, 50, 20.0, True)
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.savefig('tsne.png')
pylab.show()


# In[46]:


print("Run symmetric sne")
Y, P, Q, all_C = tsne(X, 2, 50, 20.0, False)
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.savefig('sne.png')
pylab.show()


# In[47]:


compose_gif()


# In[48]:


img_x = np.arange(2500)
plt.figure(figsize=(10,7),dpi=100,linewidth = 2)
plt.plot(img_x, Q[0], label="Q")
plt.plot(img_x, P[0], label="P")
plt.legend(loc = "best", fontsize=20)
plt.show()


# In[58]:


def check_error(all_perplexity_C):
    all_perplexity_C = np.array(all_perplexity_C)
    all_perplexity = [10, 25, 50, 100]
    
    _x = np.arange(100)
    plt.figure(figsize=(10,7),dpi=100,linewidth = 2)
    
    
    for i in range(len(all_perplexity_C)):
        plt.plot(_x, all_perplexity_C[i], label="perplexity = "+ str(all_perplexity[i]))
    
    plt.legend(loc = "best", fontsize=20)
    plt.show()    
    


# In[51]:


# test different perplexity : tsne

all_perplexity = [10, 25, 50, 100]
all_perplexity_C = []

for i in range(len(all_perplexity)):
    Y, P, Q, all_C = tsne(X, 2, 50, all_perplexity[i], True)
    all_perplexity_C.append(all_C)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.savefig('perplexity=' + str(all_perplexity[i]) + '.png')


# In[59]:


check_error(all_perplexity_C)


# In[60]:


# test different perplexity : symmetric sne

all_perplexity = [10, 25, 50, 100]
all_perplexity_C = []

for i in range(len(all_perplexity)):
    Y, P, Q, all_C = tsne(X, 2, 50, all_perplexity[i], False)
    all_perplexity_C.append(all_C)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.savefig('sne:perplexity=' + str(all_perplexity[i]) + '.png')


# In[61]:


check_error(all_perplexity_C)


# In[ ]:





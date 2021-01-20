#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


# In[2]:


def build_data():
    X = []
    Y = []
    f = open("ML_HW05-1/input.data","r")
    each_line = f.readline()
    while each_line!="":
        X_Y = each_line.strip("\n").split(" ")
        X.append(float(X_Y[0]))
        Y.append(float(X_Y[1]))
        each_line = f.readline()
    return X, Y


# In[3]:


X, Y = build_data()


# In[29]:


class GP(object):
    
    @classmethod
    def kernel_rational_quadratic(cls, x, y, l=2.47623475 , a=8.50867061, var=1.88337348):
        return var * ((1 + abs((x-y)**2) / (2 * a * (l**2))) ** (-a)) 
    
    def __init__(self, x, y, cov_f=None):
        super().__init__()
        self.x = x
        self.y = y
        self.N = len(self.x)

        self.sigma = []
        self.cov_f = cov_f if cov_f else self.kernel_rational_quadratic
        self.setup_sigma()
    
    def calculate_sigma(cls, x, cov_f):
        N = len(x)
        sigma = np.ones((N, N))
        for i in range(N):
            for j in range(N):
                sigma[i][j] = cov_f(x[i], x[j])
        return sigma

    def setup_sigma(self):
        self.sigma = self.calculate_sigma(self.x, self.cov_f)

    def predict(self, x):
        noise = 1/5
        cov =  self.cov_f(x, x) + noise
        sigma_1_2 = np.zeros((self.N, 1))
        for i in range(self.N):
            sigma_1_2[i] = self.cov_f(self.x[i], x)

        m_pre = (sigma_1_2.T * np.mat(self.sigma).I) * np.mat(self.y).T
        sigma_pre = cov - (sigma_1_2.T * np.mat(self.sigma).I) * sigma_1_2
        return m_pre, sigma_pre


# In[30]:


x = np.array(X)
y = np.array(Y)
gaus = GP(x, y)

plt.figure(figsize=(20,10))
x_guess = np.linspace(-60, 60, 400)
y_pred = np.vectorize(gaus.predict)(x_guess)

plt.scatter(x, y, c="black")
plt.plot(x_guess, y_pred[0], c="b")
plt.plot(x_guess, y_pred[0] - 1.96*np.sqrt(y_pred[1]) * 1, "r:")
plt.plot(x_guess, y_pred[0] + 1.96*np.sqrt(y_pred[1]) * 1, "r:")


# ## Minimizing negative marginal log-likelihood

# In[13]:


def rational_quadratic( x, y , l = 1, a = 1, var = 1):
    return var * ((1 + (x-y)**2 / (2 * a * (l**2))) ** (-a))


# In[17]:


def Cal_likelihood(args):
    X, Y = x, y 
    N = len(X)
    l, a, var = args
    K = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            K[i][j] = rational_quadratic(X[i], X[j], l, a, var)
     
    return -float((-1/2)*np.mat(Y)*np.mat(K).I*np.mat(Y).T - 1/2*np.log(abs(np.linalg.det(K))) - N/2*np.log(2*np.pi))


# In[18]:


x0 = np.asarray((1, 1, 1))


# In[19]:


res = minimize(Cal_likelihood, x0, method='SLSQP')
print(res.fun)
print(res.success)
print(res.x)


# In[28]:


x = np.array(X);
y = np.array(Y)
gaus = GP(x, y)

plt.figure(figsize=(20,10))
x_guess = np.linspace(-60, 60, 400)
y_pred = np.vectorize(gaus.predict)(x_guess)

plt.scatter(x, y, c="black")
plt.plot(x_guess, y_pred[0], c="b")
plt.plot(x_guess, y_pred[0] - 1.96*np.sqrt(y_pred[1]) * 1, "r:")
plt.plot(x_guess, y_pred[0] + 1.96*np.sqrt(y_pred[1]) * 1, "r:")


# ## Test different hyperaprameters

# In[54]:


flex_l = []
flex_a = []
flex_var = []

each_x = [1+0.1*i for i in range(100)]


# In[55]:


for i in range(100):
    flex_l.append(Cal_likelihood((1+0.1*i, 8.50867061, 1.88337348)))
    
for i in range(100):
    flex_a.append(Cal_likelihood((2.47623475, 1+0.1*i, 1.88337348)))
    
for i in range(100):
    flex_var.append(Cal_likelihood((2.47623475, 8.50867061, 1+0.1*i)))


# In[90]:


print(flex_l[-1])


# In[83]:


plt.plot(each_x, flex_l)
plt.title("Change L")
plt.text(1, 1366783361, 'alpha=8.50867061, variance=1.88337348', wrap=True)
plt.xlabel("L")
plt.ylabel("negative marginal log-likelihood")


# In[89]:


plt.plot(each_x, flex_a)
plt.title("Change alpha")
plt.text(2, 53.1, 'L=2.47623475, variance=1.88337348', wrap=True)
plt.xlabel("alpha")
plt.ylabel("negative marginal log-likelihood")


# In[88]:


plt.plot(each_x, flex_var)
plt.title("Change variance")
plt.text(1.5, 67, 'alpha=8.50867061, L=2.47623475', wrap=True)
plt.xlabel("Variance")
plt.ylabel("negative marginal log-likelihood")


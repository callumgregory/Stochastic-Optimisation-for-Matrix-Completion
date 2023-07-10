#!/usr/bin/env python
# coding: utf-8

# # Pre-made functions 
# 
# The functions already provided for the Coursework are provided in the cell below

# In[1]:


# Defining the run_tests() function
import pytest 
def run_tests():
    pytest.main(args=['-v','--tb=short','--color=yes'])


# In[2]:


import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fminbound

def syntheticY(n):
    """
    Synthetic rank-2 n x n matrix
    """
    U = np.vstack((np.arange(n)/n, (1-np.arange(n)/n)**2)).T
    W = np.vstack((np.arange(n)/n, (1-np.arange(n)/n)**2)).T
    return U @ W.T

def bernoulli(n,p):
    """
    Bernoulli sampling.
    Produces a set of index pairs I = {(i1,i2)} where
    i1,i2 = 0,...,n-1, and each pair is sampled with 
    probability p
    """
    I1, I2 = np.meshgrid(np.arange(n), np.arange(n))
    I1 = I1.flatten()
    I2 = I2.flatten()
    I = []
    for i1,i2 in zip(I1,I2):
        if (np.random.rand()<p):
            I.append([i1,i2])
    return np.array(I)

def Loss(U,W,Y,I):
    """
    Loss function (2), where 
    U, W are n x r factor matrices,
    Y is a n x n data matrix, 
    and I is a m x 2 matrix of index pairs
    """
    L = 0
    for i1,i2 in I:
        L = L + (U[i1] @ W[i2] - Y[i1,i2])**2
    return L / I.shape[0]

def initial(n,r):
    """
    Initial guess of the matrix factors
    """
    U0 = np.vstack((np.identity(r), np.zeros((n-r,r))))
    W0 = np.vstack((np.identity(r), np.zeros((n-r,r))))
    return U0, W0


def test_pointwise_gradient_1():
    """
    Unit tests for the pointwise gradient
    """
    Y = syntheticY(4)
    U,W = initial(4,2)
    vu = np.array([0,-1.125])
    vw = np.array([-1.125,0])
    assert np.array_equal(pointwise_gradient(U,W,Y,0,1), [vu, vw])
    
def test_pointwise_gradient_2():    
    Y = syntheticY(4)
    U,W = initial(4,2)
    vu = np.array([0,0])
    vw = np.array([-0.5,0])
    assert np.array_equal(pointwise_gradient(U,W,Y,0,2), [vu, vw])
    
def test_pointwise_gradient_3():        
    Y = syntheticY(4)
    U,W = initial(4,2)    
    vu = np.array([-0.5,0])
    vw = np.array([0,0])
    assert np.array_equal(pointwise_gradient(U,W,Y,2,0), [vu, vw])
    
    
def test_full_gradient_1():
    """
    Unit tests for the full gradient
    """
    Y = syntheticY(4)
    U,W = initial(4,2)
    Gu = np.array([[0,-1.125], [0,     0], [0,0], [0,0]])
    Gw = np.array([[0,     0], [-1.125,0], [0,0], [0,0]])
    assert np.array_equal(full_gradient(U,W,Y, np.array([[0,1]])), [Gu, Gw])    
    
def test_full_gradient_2():
    Y = syntheticY(4)
    U,W = initial(4,2)
    Gu = np.array([[0,    0], [-9/16,0], [0,   0], [0,0]]) 
    Gw = np.array([[0,-9/16], [0,    0], [-1/4,0], [0,0]])
    assert np.array_equal(full_gradient(U,W,Y, np.array([[1,0], [0,2]])), [Gu, Gw])    
    
def test_full_gradient_3():
    Y = syntheticY(4)
    U,W = initial(4,2)
    Gu = np.array([[0,-9/32], [-9/32,0], [-1/8,0], [0,0]]) 
    Gw = np.array([[0,-9/32], [-9/32,0], [0,   0], [0,0]])
    assert np.array_equal(full_gradient(U,W,Y, np.array([[1,0], [0,1], [2,0], [0,0]])), [Gu, Gw])    


# # Question 1 - Pointwise Loss Gradient
# 

# In[3]:


def pointwise_gradient(U,W,Y,i1,i2):
    "This function produces the vectors vu and vw stated as in (5) and (6)"
    # Computing the i1-th row of V^u
    vu = 2*(U[i1,:]@W[i2,:].T-Y[i1,i2])*W[i2,:]
    # Computing the i2-th row of V^w
    vw = 2*(U[i1,:]@W[i2,:].T-Y[i1,i2])*U[i1,:]
    return vu, vw


# # Question 2 - Full Gradient
# 
# We now implement Algorithm 2 as denoted in the Coursework 

# In[4]:


def full_gradient(U,W,Y,I):
    "This function works out the full gradient of as in Algorithm 2"
    # Defining the sizes of the matrices
    n = U.shape[0]
    r = U.shape[1]
    m = I.shape[0]
    # Initialising G^u, G^w = 0
    Gu = np.zeros((n,r))
    Gw = np.zeros((n,r))
    for i in range(m): 
        # Computing vu and vw as in Q1
        [vu,vw] = pointwise_gradient(U,W,Y,I[i,0],I[i,1])
        # Incrementing i1-th row of Gu
        Gu[I[i,0],:] = Gu[I[i,0],:] + vu
        # Incrementing i2-th row of Gw
        Gw[I[i,1],:] = Gw[I[i,1],:] + vw
    Gu = Gu/m
    Gw = Gw/m
    return Gu, Gw


# Checking that the `pointwise_gradient` function and the `full_gradient` function produce correct matrices by running the test functions provided

# In[5]:


run_tests()


# # Question 3 - Gradient Descent
# 
# We now implement Algorithm 3 to compute the Gradient Descent of Matrix Completion
# 
# **Note:** I have put a restriction on the growth of the gradient at each iteration to prevent the exploding gradient problem which leads to overflow 

# In[6]:


def gd(U0,W0,Y,I,Itest,t=1,eps=1e-6,K=100):
    # Initialising U=U0, W=W0, and vectors to store the test loss and iteration number
    U = U0
    W = W0
    LossVec = []
    iteration = []
    for k in range(1,K+1):
        # Case 1 - if the loss at this iteration is below the stopping tolerance
        if (Loss(U,W,Y,Itest)<eps):
            print("When t="+str(t)+" the test loss converged in "+str(k)+" iterations")
            print("When t="+str(t)+" the test loss at iteration "+str(k)+" is "+str(Loss(U,W,Y,Itest))+"")
            break
        # Case 2 - if the loss doesn't go below the tolerance level after K iterations 
        elif (Loss(U,W,Y,Itest)>eps and k==K):
            print("When t="+str(t)+" the test loss did not converge to the desired accuracy in "+str(k)+" iterations")
            print("When t="+str(t)+" the test loss at iteration "+str(k)+" is "+str(Loss(U,W,Y,Itest))+"")
            break
        # Adding the iteration number and test loss to the required vectors
        iteration.append(k)
        LossVec.append(Loss(U,W,Y,Itest))
        # Computing Gu and Gw as in Algorithm 2
        [Gu,Gw] = full_gradient(U,W,Y,I)
        # If the gradient grows too quickly (rate of 10^3 or higher) 
        # we get exploding gradients and the test loss won't converge
        if (np.linalg.norm(Gu)>1e3 or np.linalg.norm(Gw)>1e3):
            print("When t="+str(t)+" we get an exploding gradient and the test loss doesn't converge")
            break
        # Updating iterates
        U = U - t*Gu
        W = W - t*Gw 
        
    # Plotting the test loss in the logarithmic scale as a function of the iteration number
    plt.plot(iteration,LossVec, label = "t="+str(t)+"")
    plt.title('Plotting the test loss $L_{\mathcal{I}_{Test},Y}$(U,W) on the \n logarithmic scale as a function of the iteration number k')
    plt.xlabel('Iteration Number, k')
    plt.ylabel('$L_{\mathcal{I}_{Test},Y}$(U,W) on the logarithmic scale')
    plt.yscale('log')
    plt.legend()
    return U,W


# # Question 4 - Testing GD on full data
# 
# **Note:** For each run, I have made two plots. One plot contains all the values of $t$ that give overflow errors and the other plot contains the $t$ values which either converge or don't cause overflow errors.

# In[7]:


# Creating a known rank-2 matrix Y
n = 32
Y = syntheticY(n)

# Initialising random seed for comparison to PDF
np.random.seed(0) 

# Producing Iext 
p = 1
Iext = bernoulli(n,p)

# Initialising U0 and W0
r = 2
[U0,W0] = initial(n,r)

# Creating the training and test sets
from sklearn.model_selection import train_test_split
[I1,Itest1] = train_test_split(Iext, test_size=20)

# Creating the desired learning rate values 1,2,4,8,16,32
t = np.array([1,2,4,8,16,32])

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
U1 = [] 
W1 = []

# Running the gd function for the above t values
for j in t:
    [U, W] = gd(U0,W0,Y,I1,Itest1,t=j,eps=1e-8,K=5000)
    U1.append(U)
    W1.append(W)
plt.show()

# Running the gd function when t=64
t = 64 
[U, W] = gd(U0,W0,Y,I1,Itest1,t,eps=1e-8,K=5000)
U1.append(U)
W1.append(W)


# Below, 3 re-runs are computed to observe the reproducibility of the results from above for different training and test sets.

# In[8]:


# Creating new training and test sets for Re-run 1
[I2,Itest2] = train_test_split(Iext, test_size=20)

# Creating the desired learning rate values 1,2,4,8,16,32
t = np.array([1,2,4,8,16,32])

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
U2 = [] 
W2 = []

# Running the gd function for the above t values
for j in t:
    [U, W] = gd(U0,W0,Y,I2,Itest2,t=j,eps=1e-8,K=5000)
    U2.append(U)
    W2.append(W)
plt.show()

# Running the gd function when t=64
t = 64 
[U, W] = gd(U0,W0,Y,I2,Itest2,t,eps=1e-8,K=5000)
U2.append(U)
W2.append(W)


# In[9]:


# Creating new training and test sets for Re-run 2
[I3,Itest3] = train_test_split(Iext, test_size=20)

# Creating the desired learning rate values 1,2,4,8,16,32
t = np.array([1,2,4,8,16,32])

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
U3 = []
W3 = []

# Running the gd function for the above t values
for j in t:
    [U, W] = gd(U0,W0,Y,I3,Itest3,t=j,eps=1e-8,K=5000)
    U3.append(U)
    W3.append(W)
plt.show()

# Running the gd function when t=64
t = 64 
[U, W] = gd(U0,W0,Y,I3,Itest3,t,eps=1e-8,K=5000)
U3.append(U)
W3.append(W)


# In[10]:


# Creating new training and test sets for Re-run 3
[I4,Itest4] = train_test_split(Iext, test_size=20)

# Creating the desired learning rate values 1,2,4,8,16,32
t = np.array([1,2,4,8,16,32])

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
U4 = []
W4 = []

# Running the gd function for the above t values
for j in t:
    [U, W] = gd(U0,W0,Y,I4,Itest4,t=j,eps=1e-8,K=5000)
    U4.append(U)
    W4.append(W)
plt.show()

# Running the gd function when t=64
t = 64 
[U, W] = gd(U0,W0,Y,I4,Itest4,t,eps=1e-8,K=5000)
U4.append(U)
W4.append(W)


# **Note**: To present the relative errors in a table, I installed tabulate using `pip install tabulate`.

# In[11]:


# Working out the relative error for each value of t
relative_error = []
for i in range(7): 
    U = (1/4)*(U1[i] + U2[i] + U3[i] + U4[i])
    W = (1/4)*(W1[i] + W2[i] + W3[i] + W4[i])
    relative_error.append((np.linalg.norm(U@W.T-Y, 'fro'))/(np.linalg.norm(Y, 'fro')))

# Creating a table to store the data above
t = np.array([1,2,4,8,16,32,64])
data = np.array([t,relative_error])
data = data.T
col_names = ["t", "Relative Error"]
from tabulate import tabulate
print(tabulate(data, headers=col_names, tablefmt="heavy_outline", numalign='center', stralign='center')) 


# # Question 5 - Stochastic Average Gradient
# 
# We now implement Algorithm 4 - the Stochastic Average Gradient Descent Method for Matrix Completion
# 
# **Note:** I have put a restriction on the growth of the gradient at each iteration to prevent the exploding gradient problem which leads to overflow 

# In[12]:


def sag(U0,W0,Y,I,Itest,t=1,eps=1e-6,K=100): 
    # Extracting the sizes of the rows/columns for the training set and the initial guesses
    n = U0.shape[0]
    r = U0.shape[1]
    m = I.shape[0]
    # Initialising U0, W0, Vtildeu, Vtildew, Gtildeu, Gtildew and vectors for the test loss and iteration number
    U = U0 
    W = W0
    Vtildeu = np.zeros((m,r))
    Vtildew = np.zeros((m,r))
    Gtildeu = np.zeros((n,r))
    Gtildew = np.zeros((n,r))
    LossVec = []
    iteration = []
    for j in range(1, K+1): 
        # Case 1 - if the loss at this iteration is below the stopping tolerance
        if (Loss(U,W,Y,Itest)<eps):
            print("When t="+str(t)+" the test loss converged in "+str(j)+" iterations")
            print("When t="+str(t)+" the test loss at iteration "+str(j)+" is "+str(Loss(U,W,Y,Itest))+"")
            break
        # Case 2 - if the loss doesn't go below the tolerance level after K iterations
        elif (Loss(U,W,Y,Itest)>eps and j==K):
            print("When t="+str(t)+" the test loss did not converge to the desired accuracy in "+str(j)+" iterations.")
            print("When t="+str(t)+" the test loss at iteration "+str(j)+" is "+str(Loss(U,W,Y,Itest))+"")
            break
        # Adding the iteration number and test loss to the required vectors
        iteration.append(j)
        LossVec.append(Loss(U,W,Y,Itest))
        # Sampling 0,1,...,m-1 uniformly at random
        i = np.random.randint(m)
        # Taking the i-th pair from I
        [i1,i2] = I[i,:]
        # Subtracting the old gradients
        Gtildeu[i1,:] = Gtildeu[i1,:] - (1/m)*Vtildeu[i,:]
        Gtildew[i2,:] = Gtildew[i2,:] - (1/m)*Vtildew[i,:]
        # Computing Vtildeui and Vtildewi
        [Vtildeu[i,:], Vtildew[i,:]] = pointwise_gradient(U,W,Y,i1,i2)
        # Adding the new gradients
        Gtildeu[i1,:] = Gtildeu[i1,:] + (1/m)*Vtildeu[i,:]
        Gtildew[i2,:] = Gtildew[i2,:] + (1/m)*Vtildew[i,:] 
        # If the gradient grows too quickly (order of 10^3 or higher) 
        # we get exploding gradients and the test loss won't converge
        if (np.linalg.norm(Gtildeu)>1e3 or np.linalg.norm(Gtildew)>1e3):
            print("When t="+str(t)+" we get an exploding gradient and the test loss doesn't converge")
            break
        # Updating iterates 
        U = U - t*Gtildeu
        W = W - t*Gtildew
    # Plotting the test loss in the logarithmic scale as a function of the iteration number
    plt.plot(iteration, LossVec, label = "t="+str(t)+"")
    plt.title('Plotting the test loss $L_{\mathcal{I}_{Test},Y}$(U,W) on the \n logarithmic scale as a function of the iteration number k')
    plt.xlabel('Iteration Number, k')
    plt.ylabel('$L_{\mathcal{I}_{Test},Y}$(U,W) on the logarithmic scale')
    plt.yscale('log')
    plt.legend()
    return U, W


# # Question 6 - Testing SAG on full data
# 
# **Note:** 
# - For each run, I have made two plots. One plot contains all the values of $t$ that give overflow errors and the other plot contains the $t$ values which either converge or don't cause overflow errors. 
# 
# - The 4 training and test set pairings as in Q4 have been utilized below.

# In[13]:


# Initialising random seed for comparison to PDF
np.random.seed(0)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
U1 = []
W1 = []

# Creating the learning rate values 1,1/2
t = np.array([1,1/2])

# Running the sag function for the above t values
for j in t:
    [U,W] = sag(U0,W0,Y,I1,Itest1,t=j,eps=1e-8,K=50000)
    U1.append(U)
    W1.append(W)
plt.show()

# Creating the learning rate values 1/4,1/8,1/16,1/32,1/64
t = np.array([1/4,1/8,1/16,1/32,1/64])

# Running the sag function for the above t values
for j in t:
    [U,W] = sag(U0,W0,Y,I1,Itest1,t=j,eps=1e-8,K=50000)
    U1.append(U)
    W1.append(W)


# In[14]:


# Initialising random seed for comparison to PDF
np.random.seed(0)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
U2 = []
W2 = []

# Creating the learning rate values 1,1/2
t = np.array([1,1/2])

# Running the sag function for the above t values
for j in t:
    [U,W] = sag(U0,W0,Y,I2,Itest2,t=j,eps=1e-8,K=50000)
    U2.append(U)
    W2.append(W)
plt.show()

# Creating the learning rate values 1/4,1/8,1/16,1/32,1/64
t = np.array([1/4,1/8,1/16,1/32,1/64])

# Running the sag function for the above t values
for j in t:
    [U,W] = sag(U0,W0,Y,I2,Itest2,t=j,eps=1e-8,K=50000)
    U2.append(U)
    W2.append(W)


# In[15]:


# Initialising random seed for comparison to PDF
np.random.seed(1)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
U3 = []
W3 = []

# Creating the learning rate values 1,1/2
t = np.array([1,1/2])

# Running the sag function for the above t values
for j in t:
    [U,W] = sag(U0,W0,Y,I3,Itest3,t=j,eps=1e-8,K=50000)
    U3.append(U)
    W3.append(W)
plt.show()

# Creating the learning rate values 1/4,1/8,1/16,1/32,1/64
t = np.array([1/4,1/8,1/16,1/32,1/64])

# Running the sag function for the above t values
for j in t:
    [U,W] = sag(U0,W0,Y,I3,Itest3,t=j,eps=1e-8,K=50000)
    U3.append(U)
    W3.append(W)


# In[16]:


# Initialising random seed for comparison to PDF
np.random.seed(2)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
U4 = []
W4 = []

# Creating the learning rate values 1,1/2
t = np.array([1,1/2])

# Running the sag function for the above t values
for j in t:
    [U,W] = sag(U0,W0,Y,I4,Itest4,t=j,eps=1e-8,K=50000)
    U4.append(U)
    W4.append(W)
plt.show()

# Creating the learning rate values 1/4,1/8,1/16,1/32,1/64
t = np.array([1/4,1/8,1/16,1/32,1/64])

# Running the sag function for the above t values
for j in t:
    [U,W] = sag(U0,W0,Y,I4,Itest4,t=j,eps=1e-8,K=50000)
    U4.append(U)
    W4.append(W)


# **Note**: To present the relative errors in a table, I installed tabulate using `pip install tabulate`.

# In[17]:


# Working out the relative error for each value of t
relative_error = []
for i in range(7): 
    U = (1/4)*(U1[i] + U2[i] + U3[i] + U4[i])
    W = (1/4)*(W1[i] + W2[i] + W3[i] + W4[i])
    relative_error.append((np.linalg.norm(U@W.T-Y, 'fro'))/(np.linalg.norm(Y, 'fro')))

# Creating a table to store the data above
t = np.array([1,1/2,1/4,1/8,1/16,1/32,1/64])
data = np.array([t,relative_error])
data = data.T
col_names = ["t", "Relative Error"]
from tabulate import tabulate
print(tabulate(data, headers=col_names, tablefmt="heavy_outline", numalign='center', stralign='center'))


# # Question 7 - Tests with real data
# 
# Below, I have generated: 
# 
# - $I_{ext}$ of undersampling ratio $p = \frac{1}{3}$ 
# 
# - $I_{test}$ with $\mu = 20$ index pairs 
# 
# - $I = I_{ext} \setminus I_{test}$ 
# 
# - the initial guesses $U_0$, $W_0\ \in\mathbb{R}^{n\times r}$ with $r = 4$
# 
# as well as loading the matrix $Y$ provided in CW.npz

# In[18]:


# Initialising random seed for comparison to PDF
np.random.seed(10)

# Loading the given data
data = np.load('CW.npz')
# Extracting the matrix Y
Y = data['Y']

# Producing Iext 
n = Y.shape[0]
p = 1/3
Iext = bernoulli(n,p)

# Initialising U0 and W0
r = 4
[U0,W0] = initial(n,r)

# Creating the training and test sets
[I,Itest] = train_test_split(Iext, test_size=20)


# **Note:** As in previous questions, I have plotted the learning rates that cause overflow errors seperately to those that either converge or don't converge but don't cause overflow errors.

# In[19]:


# Initialising random seed for comparison to PDF
np.random.seed(10)

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
UGD = []
WGD = []

# Running the gd function with the downloaded Y with eps=1e-8 and K=5000
for j in np.array([1,2,4,8,16,32]):
    [U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=5000)
    UGD.append(U)
    WGD.append(W)
plt.show()

j = 64
[U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=5000)
UGD.append(U)
WGD.append(W)


# In[20]:


# Initialising random seed for comparison to PDF
np.random.seed(10)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
USAG = []
WSAG = []

# Running the sag function with the downloaded Y with eps=1e-8 and K=50000
for j in np.array([1,1/2,1/4,1/8]):
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=50000)
    USAG.append(U)
    WSAG.append(W)
plt.show()

for j in np.array([1/16,1/32,1/64]): 
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=50000)
    USAG.append(U)
    WSAG.append(W)


# From the results above, the minimum loss is $L_{low}=0.0019031114196868637$. Now, I set $\epsilon = 2L_{Low}$ and attempt to find the learning rate $t$ which gives fastest convergence.

# In[21]:


Llow = 0.0019031114196868637

# Initialising random seed for comparison to PDF
np.random.seed(11)

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
UGD2 = [] 
WGD2 = []

# The first run of gd using the desired tolerance as eps=2*Llow
for j in np.array([1,2,4,8,16,32]):
    [U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=5000)
    UGD2.append(U)
    WGD2.append(W)
plt.show()

j=64
[U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=5000)
UGD2.append(U)
WGD2.append(W)


# For the Gradient Descent algorithm with $\epsilon = 2L_{Low}$, we see that $t=16$ converges the fastest (in $k=459$ iterations)

# In[22]:


# Initialising random seed for comparison to PDF
np.random.seed(11)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
USAG2 = [] 
WSAG2 = []

# The first run of sag using the desired tolerance as eps=2*Llow
for j in np.array([1,1/2,1/4,1/8]):
    # Had to change the number of iterations since nothing converged when K=50000
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=200000)
    USAG2.append(U)
    WSAG2.append(W)
plt.show()

for j in np.array([1/16,1/32,1/64]):
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=200000)
    USAG2.append(U)
    WSAG2.append(W)


# For the Stochastic Average Gradient Descent algorithm with $\epsilon = 2L_{Low}$, we see that $t=\frac{1}{16}$ converges the fastest (in $k=67328$ iterations)

# We run the tests a few times to test for reproducibility:
# 
# **Re-run 1**:

# In[23]:


# Initialising random seed for comparison to PDF
np.random.seed(12)

# Creating the training and test sets for the first re-run
[I,Itest] = train_test_split(Iext, test_size=20)

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
UGDR1 = []
WGDR1 = []

# Running the gd function with the downloaded Y with eps=1e-8 and K=5000
for j in np.array([1,2,4,8,16,32]):
    [U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=5000)
    UGDR1.append(U)
    WGDR1.append(W)
plt.show()

j = 64
[U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=5000)
UGDR1.append(U)
WGDR1.append(W)


# In[24]:


# Initialising random seed for comparison to PDF
np.random.seed(12)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
USAGR1 = []
WSAGR1 = []

# Running the sag function with the downloaded Y with eps=1e-8 and K=50000
for j in np.array([1,1/2,1/4,1/8]):
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=50000)
    USAGR1.append(U)
    WSAGR1.append(W)
plt.show()

for j in np.array([1/16,1/32,1/64]): 
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=50000)
    USAGR1.append(U)
    WSAGR1.append(W)


# From the results in Re-run 1, the minimum loss is $L_{low}=0.0022083155178672723$. Now, I set $\epsilon = 2L_{Low}$ and attempt to find the learning rate $t$ which gives fastest convergence.

# In[25]:


Llow = 0.0022083155178672723

# Initialising random seed for comparison to PDF
np.random.seed(13)

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
UGDR12 = []
WGDR12 = []

# Running gd using the desired tolerance as eps=2*Llow
for j in np.array([1,2,4,8,16,32]):
    [U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=5000)
    UGDR12.append(U)
    WGDR12.append(W)
plt.show()

j = 64
[U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=5000)
UGDR12.append(U)
WGDR12.append(W)


# For the Gradient Descent algorithm with $\epsilon = 2L_{Low}$, we see that $t=16$ converges the fastest (in $k=432$ iterations)

# In[26]:


# Initialising random seed for comparison to PDF
np.random.seed(13)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
USAGR12 = []
WSAGR12 = []

# Running sag using the desired tolerance as eps=2*Llow
for j in np.array([1,1/2,1/4,1/8]):
    # Had to change the number of iterations since nothing converged when K=50000
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=200000)
    USAGR12.append(U)
    WSAGR12.append(W)
plt.show()

for j in np.array([1/16,1/32,1/64]): 
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=200000)
    USAGR12.append(U)
    WSAGR12.append(W)


# For the Stochastic Average Gradient Descent algorithm with $\epsilon = 2L_{Low}$, we see that $t=\frac{1}{16}$ converges the fastest (in $k=124536$ iterations)

# **Re-run 2**:

# In[27]:


# Initialising random seed for comparison to PDF
np.random.seed(14)

# Creating the training and test sets for the first re-run
[I,Itest] = train_test_split(Iext, test_size=20)

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
UGDR2 = []
WGDR2 = []

# Running the gd function with the downloaded Y with eps=1e-8 and K=5000
for j in np.array([1,2,4,8,16,32]):
    [U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=5000)
    UGDR2.append(U)
    WGDR2.append(W)
plt.show()

j = 64
[U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=5000)
UGDR2.append(U)
WGDR2.append(W)


# In[28]:


# Initialising random seed for comparison to PDF
np.random.seed(14)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
USAGR2 = []
WSAGR2 = []

# Running the sag function with the downloaded Y with eps=1e-8 and K=50000
for j in np.array([1,1/2,1/4,1/8]):
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=50000)
    USAGR2.append(U)
    WSAGR2.append(W)
plt.show()

for j in np.array([1/16,1/32,1/64]): 
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=1e-8,K=50000)
    USAGR2.append(U)
    WSAGR2.append(W)


# From the results in re-run 2, the minimum loss is $L_{low}=0.0036326306804015766$. Now, I set $\epsilon = 2L_{Low}$ and attempt to find the learning rate $t$ which gives fastest convergence.

# In[ ]:


Llow = 0.0036326306804015766

# Initialising random seed for comparison to PDF
np.random.seed(15)

# Creating vectors to store the final values of U and W in gd algorithm for each value of t
UGDR21 = []
WGDR21 = []

# Running gd using the desired tolerance as eps=2*Llow
for j in np.array([1,2,4,8,16,32]):
    [U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=5000)
    UGDR21.append(U)
    WGDR21.append(W)
plt.show()

j = 64
[U,W] = gd(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=5000)
UGDR21.append(U)
WGDR21.append(W)


# For the Gradient Descent algorithm with $\epsilon = 2L_{Low}$, we see that $t=16$ converges the fastest (in $k=199$ iterations)

# In[ ]:


# Initialising random seed for comparison to PDF
np.random.seed(15)

# Creating vectors to store the final values of U and W in sag algorithm for each value of t
USAGR21 = []
WSAGR21 = []

# Running sag using the desired tolerance as eps=2*Llow
for j in np.array([1,1/2,1/4,1/8]):
    # Had to change the number of iterations since nothing converged when K=50000
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=200000)
    USAGR21.append(U)
    WSAGR21.append(W)
plt.show()

for j in np.array([1/16,1/32,1/64]): 
    [U,W] = sag(U0,W0,Y,I,Itest,t=j,eps=2*Llow,K=200000)
    USAGR21.append(U)
    WSAGR21.append(W)


# For the Stochastic Average Gradient Descent algorithm with $\epsilon = 2L_{Low}$, we see that $t=\frac{1}{16}$ converges the fastest (in $k=23708$ iterations)

# Looking at my initial run and the 2 re-runs above, I use the $U$ and $W$ which correspond to the smallest test loss to produce the most accurate $UW^T$ I can. Upon observing the results above, this is seen in the initial run where $t=8$ in the Gradient Descent algorithm and this gives a loss of $0.003800354810265238$

# In[ ]:


# Plotting the actual data Y 
plt.imshow(Y)
plt.title("Plot of the actual Y")
plt.show()
# Plotting my most accurate UW^T 
plt.imshow(UGD2[3]@WGD2[3].T)
plt.title("My most accurate recreation of Y")
plt.show()


# Working out the relative error for Gradient Descent:

# In[ ]:


# Working out the relative error for each value of t for GD
relative_error = []
for i in range(7): 
    U = (1/3)*(UGD2[i] + UGDR12[i] + UGDR21[i])
    W = (1/3)*(WGD2[i] + WGDR12[i] + WGDR21[i])
    relative_error.append((np.linalg.norm(U@W.T-Y, 'fro'))/(np.linalg.norm(Y, 'fro')))

# Creating a table to store the data above
t = np.array([1,2,4,8,16,32,64])
data = np.array([t,relative_error])
data = data.T
col_names = ["t", "Relative Error"]
from tabulate import tabulate
print(tabulate(data, headers=col_names, tablefmt="heavy_outline", numalign='center', stralign='center'))


# Working out the relative error for Stochastic Average Gradient Descent:

# In[ ]:


# Working out the relative error for each value of t for SAG
relative_error = []
for i in range(7): 
    U = (1/3)*(USAG2[i] + USAGR12[i] + USAGR21[i])
    W = (1/3)*(WSAG2[i] + WSAGR12[i] + WSAGR21[i])
    relative_error.append((np.linalg.norm(U@W.T-Y, 'fro'))/(np.linalg.norm(Y, 'fro')))

# Creating a table to store the data above
t = np.array([1,1/2,1/4,1/8,1/16,1/32,1/64])
data = np.array([t,relative_error])
data = data.T
col_names = ["t", "Relative Error"]
from tabulate import tabulate
print(tabulate(data, headers=col_names, tablefmt="heavy_outline", numalign='center', stralign='center'))


# In[ ]:





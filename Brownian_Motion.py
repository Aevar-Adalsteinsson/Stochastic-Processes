import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

def BM(n,T,mu,sigma):
    dt = T/n
    N_n = np.sqrt(dt) * np.random.normal(0, sigma, n)
    fin_var = mu*dt*np.arange(n+1)
    W = np.insert(np.cumsum(N_n),0,0) + fin_var
    return W

def max_process(A):
    n = len(A)
    max_val = -math.inf
    for a in A:
        if a > max_val:
            max_val = a
    return(max_val)
def min_process(A):
    n = len(A)
    min_val = math.inf
    for a in A:
        if a < min_val:
            min_val = a
    return(min_val)
def max_diff_process(A):
    n = len(A)
    max_val = -math.inf
    min_val = math.inf
    for a in A:
        if a > max_val:
            max_val = a
        if a < min_val:
            min_val = a
    return(max_val - min_val)

if __name__ == '__main__':
    #Properties of the brownian motion
    
    #Maximum of the Brownian Motion at time t is equal to the Half-normal distribution with parameter t
    #The half-normal distribution is defined to be the absolute value of the normal distribution
    #i.e X = N(0,sigma^2), then Y = |X| is a Half-normal distribution with parameter sigma^2 
    mu = 0
    sigma = 1
    t = 1
    m = 10000 #number of iterations
    n = 1000 #number of segments of the brownian motion
    M_t = np.zeros(m)
    Half_normal = np.abs(np.random.normal(mu,t,m))
    for i in np.arange(m):
        brownian = BM(n,t,mu,sigma)
        M_t[i] = max_process(brownian)
    
    M_t = np.sort(M_t)
    Half_normal = np.sort(Half_normal)

    plt.plot(M_t)
    plt.plot(Half_normal)
    plt.show()
    plt.plot(M_t-Half_normal)
    plt.show()
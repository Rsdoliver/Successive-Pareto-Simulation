# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:55:20 2018

@author: Adriano
"""
# input:M,    S,   T
#       Mean, STD, RV Type
# output: P (pdf parameters)

from scipy import stats
import scipy as sp
import scipy.special
from scipy.optimize import fsolve
import numpy as np
import math

pi = 3.14159
gamma = 0.5772 #contante de euler 
def pdf_parameters(M,S,T):
    def k_weibull(k):
        gamma1 = scipy.special.gamma(1 + 1/k)
        gamma2 = scipy.special.gamma(1 + 2/k)
        return np.sqrt(gamma2/gamma1**2 - 1) - (S/M)
    return {
      'norm': lambda x,y:[M,S],
      'lognorm': lambda x,y:[np.exp(logn_p(M,S)[0]),logn_p(M,S)[1]],
      'gamma': lambda x,y:[M**2/S**2,S**2/M],
      'expon': lambda x,y:[M],
      'gumbel': lambda x,y:[M - gamma*(S*(6)**(1/2))/(pi),(S*(6)**(1/2))/(pi)],
      'weibull': lambda x,y:[fsolve(k_weibull,1.5)[0],(M/(scipy.special.gamma(1+1/(fsolve(k_weibull,[1.5])))))[0]]
      }[T](M,S)
      
def logn_p(m,s):
    mu = np.log((m**2)/(s**2+m**2)**.5);
    sigma = (np.log(s**2/(m**2)+1))**.5;
    return mu,sigma

def cdf(T,X,P):
    return {
      'norm': lambda x,y:stats.norm.cdf(X,P[0],P[1]),
      'lognorm': lambda x,y:stats.lognorm.cdf(X/P[0],P[1]),
      'gamma': lambda x,y:stats.gamma.cdf(X/P[1],P[0]),
      'expon': lambda x,y:stats.expon.cdf(X/P[0],P[0]),
      'gumbel': lambda x,y:stats.gumbel_r.cdf(X,P[0],P[1]),
      'weibull': lambda x,y:stats.weibull_min.cdf(X,P[0],scale=P[1])
    }[T](X,P)
    
    
def pdf(T,X,P):
    return {
      'norm': lambda x,y:stats.norm.pdf(X,P[0],P[1]),
      'lognorm': lambda x,y:stats.lognorm.pdf(X/P[0],P[1])/P[0],
      'gamma': lambda x,y:stats.gamma.pdf(X/P[1],P[0])/P[1],
      'expon': lambda x,y:stats.expon.pdf(X/P[0])/P[0],
      'gumbel': lambda x,y:stats.gumbel_r.pdf(X,P[0],P[1]),
      'weibull': lambda x,y:stats.weibull_min.pdf(X,P[0],scale=P[1])
    }[T](X,P)

def generate_sample(T,P,N):
    return {
      'norm': lambda x:stats.norm.rvs(P[0],P[1],size=N),
      'lognorm': lambda x:stats.lognorm.rvs(P[1],scale = P[0],size = N),
      'gamma': lambda x:np.random.gamma(shape=(P[0]), scale=(P[1]), size=N),
      'gumbel': lambda x:stats.gumbel_r.rvs(loc = P[0], scale = P[1], size = N),
      'weibull': lambda x:P[1]*np.random.weibull(P[0],size=N)
    }[T](P)

# Inverse of CDF
def icdf(T, X, P):
    return {
        'norm':     lambda x, y: stats.norm.ppf(x, loc=y[0], scale=y[1]),
        'lognorm':  lambda x, y: stats.lognorm.ppf(x, s=y[1], scale=y[0]),
        'gamma':    lambda x, y: stats.gamma.ppf(x, a=y[0], scale=y[1]),
        'expon':    lambda x, y: stats.expon.ppf(x, loc=0, scale=y[0]),  # P[0] = mean = scale
        'gumbel':   lambda x, y: stats.gumbel_r.ppf(x, loc=y[0], scale=y[1]),
        'weibull':  lambda x, y: stats.weibull_min.ppf(x, c=y[0], scale=y[1])
    }[T](X, P)

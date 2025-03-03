# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 00:17:44 2024

@author: rsdol
"""

# Exemplo usando NSPS

import numpy as np
import scipy.stats as stats
from NewSPS import PSMC

#%% Variáveis de entrada

N = int(3350) # Sample size
nRV = 2 # Number of random variables
M = np.array([0,0]) # Mean vector
S = np.array([1,1]) # Standard deviation vector

#%% Funções

# Função de falha

def fun_G(X):
    G = 10 - 2*X[0] - 3*X[1];
    return G

#%% Geração de valores aleatórios

Xs = np.zeros((N,nRV))

for i in range(nRV):
    Xs[:,i] = np.array(list(np.random.normal(M[i],S[i],size=N)))


#%% Cálculo
[N_fail,F_count,direction] = PSMC(Xs,fun_G,M,S,direction=[])

PF_SMC = N_fail/N # Probabilidade de Falha

if PF_SMC == 0:
    CoV = 0;
else:
    CoV = np.sqrt((1-PF_SMC)/(PF_SMC*N)) # Coeficiente de Variação

print('\nProbabilidade de falha = {}'.format(PF_SMC))
print('Beta = {}'.format(-stats.norm.ppf(PF_SMC)))
print('Coeficiente de Variação = {}'.format(CoV))
print('Número de avaliações de G = {}'.format(F_count))
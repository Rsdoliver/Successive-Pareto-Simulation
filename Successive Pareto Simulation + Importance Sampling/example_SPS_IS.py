# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 00:17:44 2024

@author: rsdol
"""

# Successive Pareto Simulation + Importance Sampling (Prof. Renato Motta)
# Example

import numpy as np
import scipy.stats as stats
from SPS_IS import PSMC
from Distribution_class import pdf_parameters
from FORM2 import form

#%% Variáveis de entrada
PF = 0.;
aval = 0;
cov = 0.;
n = 1;
PF_i = np.zeros(n)
for k in range(n):
    N = int(10000)
    nRV = 2
    M = np.array([0.,0.])
    S = np.array([1.,1.])
    
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

    # Array of Dist. Types
    RV_type=['norm','norm'];

    #num. de variaveis aleatorias
    nRV = len(M);

    P = list(range(nRV))
    for i in range(nRV):
        P[i] = pdf_parameters(M[i],S[i],RV_type[i])

    #############################
    # FORM 
    [PF_form, beta, MPP, ii] = form(M,RV_type,nRV,P,fun_G)

    [N_fail,F_count,direction,times,PF_SPSIS,CoVIS] = PSMC(Xs,fun_G,M,S,MPP,RV_type,P)
    PF = PF + PF_SPSIS
    PF_i[k] = PF_SPSIS
    aval = aval + F_count
    cov += CoVIS
    print('\nProbabilidade de falha = {}'.format(PF_SPSIS))
    print('Beta = {}'.format(-stats.norm.ppf(PF_SPSIS)))
    print('Coeficiente de Variação = {}'.format(CoVIS))
    print('Número de avaliações de G = {}'.format(F_count))
    
print("\nPROBABILIDADE DE FALHA FINAL:",PF/n)
print("\nNe FINAL:",aval/n)
print("\nCoV FINAL calculado:",cov/n)
print("\nCoV FINAL estimado:",np.std(PF_i)/np.mean(PF_i))

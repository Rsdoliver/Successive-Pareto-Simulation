# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 00:17:44 2024

@author: rsdol
"""

# Exemplo usando Simulação Sucessiva de Pareto
# Problema 2 de Santos et al. (2012)
# S.R. Santos, L.C. Matioli, A.T. Beck, New optimization algorithms for structural reliability, 
# Comput. Model. Eng. Sci. 83 (2012) 23–55. https://doi.org/10.3970/cmes.2012.083.023.

import numpy as np
import scipy.stats as stats
from SPS import PSMC
from Distribution_class import pdf_parameters, generate_sample

#%% Variáveis de entrada
PF = 0;
aval = 0;
n = 1;
for i in range(n):
    N = int(3350) # Sample size
    nRV = 2 # Number of random variables
    M = np.array([0.,0.]) # Mean vector
    S = np.array([1.,1.]) # Standard deviation vector
    
    # Array of Dist. Types
    RV_type=['norm','norm'];
    
    #%% Funções
    
    # Função de falha
    
    def fun_G(X):
        G = -0.5*(X[0]-X[1])**2 - (X[0] + X[1])/np.sqrt(2) + 3;
        return G
    
    #%% Geração de valores aleatórios (amostra)
    Xs = np.zeros((N,nRV))

    # Parâmetros

    P = list(range(nRV))
    for i in range(nRV):
        P[i] = pdf_parameters(M[i],S[i],RV_type[i])

    # Amostra

    for i in range(nRV):
        Xs[:,i] = generate_sample(RV_type[i],P[i],N)

    np.save('Xspipe',Xs) # Salvar amostra para comparação com MC Tradicional (opcional)
    
    #%% Cálculo

    [N_fail,F_count,direction] = PSMC(Xs,fun_G,M,S)
    
    PF_SMC = N_fail/N # Probabilidade de Falha

    if PF_SMC == 0:
        CoV = 0;
    else:
        CoV = np.sqrt((1-PF_SMC)/(PF_SMC*N)) # Coeficiente de Variação
    
    PF = PF + PF_SMC
    aval = aval + F_count
    print('\nProbabilidade de falha = {}'.format(PF_SMC))
    print('Beta = {}'.format(-stats.norm.ppf(PF_SMC)))
    print('Coeficiente de Variação = {}'.format(CoV))
    print('Número de avaliações de G = {}'.format(F_count))

print("\nPROBABILIDADE DE FALHA FINAL:",PF/n)
print("\nNe FINAL:",aval/n)

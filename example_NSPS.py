# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 00:17:44 2024

@author: rsdol
"""

# Exemplo da nova Simulação Sucessiva de Pareto (Prof. Renato Motta)

import numpy as np
import scipy.stats as stats
from NewSPS import PSMC
import time
from Distribution_class import pdf_parameters, generate_sample

#%% Variáveis de entrada

N = int(1e4)
nRV = 2
M = np.array([0,0])
S = np.array([1,1])

# Array of Dist. Types
RV_type=['norm','norm'];

#%% Funções

# Função de falha

def fun_G(X):
    G = 10 - 2*X[0] - 3*X[1];
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
    
np.save('XsMC',Xs) # Salvar amostra para comparação com MC Tradicional (opcional)


#%% Cálculo
start = time.time()
[N_fail,F_count,direction,times] = PSMC(Xs,fun_G,M,S,RV_type,P,direction=[],width=1,max_fail_stack=5,plot=1)
end = time.time()
print("Total time (s): ",end-start)
print("Filtering time (s): ",times)

PF_SMC = N_fail/N # Probabilidade de Falha

if PF_SMC == 0:
    CoV = 0;
else:
    CoV = np.sqrt((1-PF_SMC)/(PF_SMC*N)) # Coeficiente de Variação

print('\nProbabilidade de falha = {}'.format(PF_SMC))
print('Beta = {}'.format(-stats.norm.ppf(PF_SMC)))
print('Coeficiente de Variação = {}'.format(CoV))
print('Número de avaliações de G = {}'.format(F_count))

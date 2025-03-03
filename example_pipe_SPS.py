# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 09:34:17 2021

@author: Rodrigo
"""

# Monte Carlo Seletivo (Prof. Renato Motta)
# ARQUIVO DE ENTRADA

import numpy as np
import scipy.stats as stats
from SPS import PSMC
from Distribution_class import pdf_parameters, generate_sample

#%% Variáveis de entrada

N = 618400      # Tamanho da amostra do MC
nRV = 6            # Número de variáveis aleatórias

De = 406.4   # Diâmetro externo (mm)
t = 12.7      # Espessura do duto (mm)
ld = 173.9    # Comprimento do defeito (mm)
d = 2.17     # Profundidade do defeito (mm)
Pint = 17.28  # Pressão interna de operação (MPa)
sigmay = 410.7 # Tensão de escoamento (MPa)
M = [De,t,ld,d,Pint,sigmay];    # Vetor com as médias
S = [0.41,0.13,87,1.09,1.2,32.86];     # Vetor com os desvios-padrão

# Array of Dist. Types
RV_type=['norm','norm','weibull','weibull','gumbel','lognorm'];

#%% Funções

# Função de falha

def fun_G(X):
    m = np.sqrt(1+0.8*((X[2]/X[0])**2)*(X[0]/X[1]))
    G = ((1.1*X[5]*2*X[1])/X[0])*((1-(2/3)*(X[3]/X[1]))/(1-(2/3)*(X[3]/X[1])*m**(-1)))-X[4];
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






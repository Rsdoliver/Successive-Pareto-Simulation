# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:31:32 2023

@author: rsdol
"""

# Successive Pareto Simulation + Importance Sampling (Prof. Renato Motta - UFPE)

import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import matplotlib
import scipy
import Distribution_class as DC
from scipy import stats
            
def weight(X,center):
    [N,nv] = [np.size(X,0),np.size(X,1)]
    
    pdf_target = np.zeros((N,nv))
    pdf_reference = np.zeros((N,nv))
    
    # Original PDF in standard Gaussian space (mean 0, std 1)
    for i in range(nv):
        pdf_target[:,i] = scipy.stats.norm.pdf(X[:,i], loc=0, scale=1)
    pdf_target = np.prod(pdf_target, axis=1)
    
    # Sampling function in standard Gaussian space (mean at center, std 1)
    for i in range(nv):
        pdf_reference[:,i] = scipy.stats.norm.pdf(X[:,i], loc=center[i], scale=1)
    pdf_reference = np.prod(pdf_reference, axis=1)

    # Sampling weights
    w = pdf_target/pdf_reference
    
    return w

# Pareto front
def domination_filter(X,direction):
    
    start = time.time()
    pdominant = np.arange(X.shape[0])
    next_point_index = 0  # Next index in the pdominant array to search for
    while next_point_index<len(X):
        nondominated_point_mask = np.any(X>X[next_point_index], axis=1) # Search for non-dominated points
        nondominated_point_mask[next_point_index] = True # The point used as reference is non-dominated
        pdominant = pdominant[nondominated_point_mask] # Remove dominated points
        X = X[nondominated_point_mask] # Remove dominated points
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1 # Next reference point
    end = time.time()
    time1 = end - start
    return pdominant, time1

# Failure direction
def faildir(M,S,fun_G,nv):
    
    X = M
    dx = S
    G = fun_G(X)

    # Finite diferences (right) at the mean
    dGdx = np.arange(float(nv))
    for i in range(nv):
        x1 = copy.deepcopy(X)
        x1[i] = x1[i]+dx[i]
        G1 = fun_G(x1)
        dGdx[i]=(G1-G)/dx[i]

    direction = -np.sign(dGdx) # Failure direction
    
    return direction

def x_to_u(X,RV_type,P,central_point=0):
    
    X_normal = X*0
    Seq = X*0
    Meq = X*0
    # Normal Equivalent Distribution
    for j in range (np.size(X,0)):
        for i in range(np.size(X,1)):
            #print(RV_type[i],X[i],P[i][:])
            Px=DC.cdf(RV_type[i],X[j,i],P[i][:])
            px=DC.pdf(RV_type[i],X[j,i],P[i][:])
            if px<1e-25:
                px = 1e-25
            if Px<1e-25:
                Px = 1e-25
            X_normal[j,i]=stats.norm.ppf(Px,central_point[i],1)
            Seq[j,i]=stats.norm.pdf(X_normal[j,i])/px
            Meq[j,i]=X[j,i]-X_normal[j,i]*Seq[j,i]
    
    return X_normal, Meq, Seq

# def u_to_x(X_normal,Meq,Seq):
#     # Inicializa a matriz X com zeros, com o mesmo shape da matriz normalizada
#     X = X_normal*Seq+Meq

#     return X

def u_to_x(X_normal, RV_type, P):
    # Inicializa a matriz X com zeros, com o mesmo shape da matriz normalizada
    X = np.zeros_like(X_normal)

    for i in range(X_normal.shape[0]):
            # Probabilidade acumulada no espaço normal padrão
            u_cdf = stats.norm.cdf(X_normal[i])

            # Garante que o valor fique dentro de uma faixa numérica válida
            u_cdf = np.clip(u_cdf, 1e-10, 1 - 1e-10)

            # Usa a inversa da CDF da distribuição original
            X[i] = DC.icdf(RV_type[i], u_cdf, P[i][:])  # ppf = percent point function (CDF inversa)

    return X

# Selective Monte Carlo
def PSMC(X,fun_G,M,S,MPP,RV_type,P,direction=[],plot=[]):
    # direction = List with 1 or -1 for each RV (float)
    # plot = Set any value to generate plot figures (PNG) for problems with 2 variables
    
    # Initialize variables
    N = np.size(X,axis=0)
    Ni = N # Initial sample size
    nv = np.size(X,axis=1)
    times = 0; # Count time
    if direction == []:
        direction = faildir(M,S,fun_G,nv); # Find initial failure direction
    else:
        direction = np.array(direction) # Use given initial failure direction
    iteration = 0 # Iteration counter
    F_count = 0 # Function evaluations counter
    N_fail = 0 # Failed points counter
    weights = 0.
    weights_sq = 0.
    CoVIS = 1.
    central_point = MPP
    
    # New sample
    X = np.zeros((N,nv))
    for i in range(nv):
        X[:,i] = np.array(list(np.random.normal(central_point[i],1.,size=N)))
    
    # Sum of weights (for normalizing purpose)
    sum_w = 0.
    w = weight(X,central_point)
    sum_w = np.sum(w)
    all_weight_list = w
    
    # Selecting procedure
    while not list(X) == []:
        
        # Initialize variables
        [N,nv] = [np.size(X,0),np.size(X,1)] # N = Number of samples, nv = Number of RVs
        Xaux = X*(np.ones((N,1)) @ direction.reshape(1,nv)) # Correct signs through directions to select correctly
        pdominant, time1 = domination_filter(Xaux,direction) # Find dominant points (index)
        iteration += 1
        n_index = np.arange(X.shape[0]) # Index list of the sample
        count = 0
        gd = pdominant*0.
        
        # Failure function evaluations on dominant points
        for i in pdominant:
            gd[count] = fun_G(u_to_x(X[i,:],RV_type,P))
            count += 1
        
        # IS weights
        if len(pdominant[gd<=0]):
            w = weight(X[pdominant[gd<=0]],central_point)
            for i in range(len(w)):
                weights += w[i]
                weights_sq += w[i]**2
    
        # Function evaluations count
        F_count += count
        
        # Set with dominant failed points
        i_out = set(pdominant[gd<=0])
    
        # Failed points count
        i_fail = len(i_out)
        
        # Total number of failed points
        N_fail = N_fail + i_fail
        
        # Start filtering time measure
        start = time.time()
        
        # Safe dominant and dominated points count
        safe_out = set()
        for i in pdominant[gd>0]:
            dominate_over = np.any(Xaux>Xaux[i], axis=1)==0 # Find safe dominated points
            safe_out = safe_out | set(n_index[dominate_over]) # Remove safe dominated points
        
        # Plot (for 2 variables)
        if not plot == []:
            if nv == 2:
                safe_par = set(pdominant[gd>0])
                if iteration == 1:
                    matplotlib.rcParams['font.family'] = 'serif'
                    matplotlib.rcParams['font.serif'] = ['Times New Roman']
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(X[:,0],X[:,1],c='blue',s=3, label = 'Sample points')
                    ax.legend(loc='lower left',fontsize=14)
                    plt.xlabel('$\\it{X_{1}}$',fontsize=14)
                    plt.ylabel('$\\it{X_{2}}$',fontsize=14)
                    plt.tick_params(axis='both', which='major', labelsize=14)
                    plt.title('Original sample',fontsize=20)
                    plt.savefig('smc_sample.png', format='png', dpi=300, bbox_inches='tight')
                    plt.scatter(X[list(i_out),0],X[list(i_out),1],c='red',marker='D', label='Failed Pareto points',s=40)
                    plt.scatter(X[list(safe_par),0],X[list(safe_par),1],c='green',marker='o', label = 'Safe Pareto points',s=40)
                    plt.scatter(X[list(safe_out-safe_par),0],X[list(safe_out-safe_par),1],c='black',marker='x', label = 'Removed points',s=40)
                    plt.title('Iteration ' + str(iteration) + ' ($\\it{N_{e}}$ = ' + str(F_count)+')',fontsize=16)
                    ax.legend(loc='lower left',fontsize=14)  # Display legend
                else:
                    plt.scatter(X[list(i_out),0],X[list(i_out),1],c='red',marker='D', label='Failed Pareto points',s=60)
                    plt.scatter(X[list(safe_par),0],X[list(safe_par),1],c='green',marker='o', label = 'Safe Pareto points',s=60)
                    plt.scatter(X[list(safe_out-safe_par),0],X[list(safe_out-safe_par),1],c='black',marker='x', label = 'Removed points',s=60)
                    plt.title('Iteration ' + str(iteration) + ' ($\\it{N_{e}}$ = ' + str(F_count)+')',fontsize=20)
                    plt.xlabel('$\\it{X_{1}}$',fontsize=24)
                    plt.ylabel('$\\it{X_{2}}$',fontsize=24)
                    plt.tick_params(axis='both', which='major', labelsize=24)
                plt.savefig('smc_iter_'+str(iteration)+'.png', format='png', dpi=1200, bbox_inches='tight')
                if iteration == 1:
                    legend = ax.get_legend()  # Retrieve the existing legend
                    if legend is not None:
                        for text in legend.get_texts():
                            text.set_fontsize(18)
                if iteration == 2:
                    plt.legend().remove()
        
        # End filtering time measure
        end = time.time()
        times = times + (end - start) + time1
        
        # # Set update to include safe dominant and dominated points
        i_out |= safe_out
        
        # Remove from the sample dominant points and safe dominated points
        X = np.delete(X,(list(i_out)),axis=0)
        
        print('iteration = {}, N_fail = {}, i_fail = {}, F_count = {}, sample size = {}, Pareto points = {}'.format(iteration,N_fail,i_fail,F_count,X.shape[0],count))
    
    PF_SMC = weights/Ni # Probabilidade de Falha
    #PF_SMC = weights/sum_w
    
    if PF_SMC == 0:
        CoVIS = 0;
    else:
        CoVIS = np.sqrt((weights_sq/Ni - (PF_SMC**2))/Ni)/PF_SMC;
        # all_normalized_w = np.array(all_weight_list)/sum_w
        # all_weights_sq = np.sum(all_normalized_w**2)
        # CoVIS = np.sqrt((1-2*PF_SMC)*weights_sq/(sum_w**2) + (PF_SMC**2)*all_weights_sq)/PF_SMC
    
    print('\nProbabilidade de falha = {}'.format(PF_SMC))
    print('Beta = {}'.format(-scipy.stats.norm.ppf(PF_SMC)))
    print('Coeficiente de Variação = {}'.format(CoVIS))
    print('Número de avaliações de G = {}'.format(F_count))
    
    return [N_fail,F_count,direction,times,PF_SMC,CoVIS]
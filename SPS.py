# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:31:32 2023

@author: rsdol
"""

# Successive Pareto Simulation (Prof. Renato Motta - UFPE)

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib

# Pareto filter
def domination_filter(X,direction):

    pdominant = np.arange(X.shape[0])
    next_point_index = 0  # Next index in the pdominant array to search for
    while next_point_index<len(X):
        nondominated_point_mask = np.any(X>X[next_point_index], axis=1) # Search for non-dominated points
        nondominated_point_mask[next_point_index] = True # The point used as reference is non-dominated
        pdominant = pdominant[nondominated_point_mask]  # Remove dominated points
        X = X[nondominated_point_mask] # Remove dominated points
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1 # Next reference point

    return pdominant

# Failure direction
def faildir(M,S,fun_G,nRV):
    
    X = M
    dx = S
    G = fun_G(X)

    # Finite diferences (right) at the mean
    dGdx = np.arange(float(nRV))
    for i in range(nRV):
        x1 = copy.deepcopy(X)
        x1[i] = x1[i]+dx[i]
        G1 = fun_G(x1)
        dGdx[i]=(G1-G)/dx[i]

    direction = -np.sign(dGdx) # Failure direction
    
    return direction

# Monte Carlo Seletivo
def PSMC(X,fun_G,M,S,direction=[]):
    
    nRV = np.size(X,axis=1)
    if direction == []:
        direction = faildir(M,S,fun_G,nRV);
        
    iteration = 0 # Iteration counter
    F_count = 0 # Function evaluations counter
    N_fail = 0 # Failed points counter
    
    while not list(X) == []:
        [N,nv] = [np.size(X,0),np.size(X,1)] # N = Number of samples, nv = Number of RVs
        Xaux = X*(np.ones((N,1)) @ direction.reshape(1,nv)) # Correct signs through directions to filter correctly
        pdominant = domination_filter(Xaux,direction) # Find dominant points (index)
        iteration += 1
        n_index = np.arange(X.shape[0]) # Index list of the sample
        
        count = 0
        gd = pdominant*0.
        
        for i in pdominant:
            gd[count] = fun_G(X[i,:]) # Failure function evaluations on dominant points
            count += 1
        
        # Function evaluations count
        F_count += count
        
        # Set with dominant failed points
        i_out = set(pdominant[gd<=0])
        
        # Failed points count
        i_fail = len(i_out)
        N_fail = N_fail + i_fail # Number of failed points
        
        # Safe dominant and dominated points count
        safe_out = set()
        for i in pdominant[gd>0]:
            dominate_over = n_index[np.logical_and(np.any(Xaux>Xaux[i], axis=1)==0,np.isin(n_index,list(safe_out)) == 0)] # Find safe dominated points
            safe_out |= set(dominate_over) # Remove safe dominated points
        
        # # Plot (for 2 variables)
        if nRV == 2:
            safe_par = set(pdominant[gd>0])
            if iteration == 1:
                matplotlib.rcParams['font.family'] = 'serif'
                matplotlib.rcParams['font.serif'] = ['Times New Roman']
                plt.figure(figsize=(7, 5))
                plt.scatter(X[:,0],X[:,1],c='blue',s=8, label = 'Sample points')
                plt.legend(loc='lower left')
                plt.xlabel('$\\it{X_{1}}$')
                plt.ylabel('$\\it{X_{2}}$')
                plt.title('Original sample')
                plt.savefig('smc_sample.png', format='png', dpi=300, bbox_inches='tight')
            plt.scatter(X[list(i_out),0],X[list(i_out),1],c='red',marker='D', label='Failed Pareto points')
            plt.scatter(X[list(safe_par),0],X[list(safe_par),1],c='green',marker='o', label = 'Safe Pareto points')
            plt.scatter(X[list(safe_out-safe_par),0],X[list(safe_out-safe_par),1],c='black',marker='x', label = 'Removed points')
            plt.title('Iteration ' + str(iteration) + ' ($\\it{N_{e}}$ = ' + str(F_count)+')')
            if iteration == 1:
                plt.legend(loc='lower left')  # Display legend
            plt.savefig('smc_iter_'+str(iteration)+'.png', format='png', dpi=1200, bbox_inches='tight')
       
        
        # Set update to include safe dominant and dominated points
        i_out |= safe_out
        
        # Remove from the sample dominant points and safe dominated points
        X = np.delete(X,(list(i_out)),axis=0)
        
        #print('iteration = {}, N_fail = {}, i_fail = {}, F_count = {}'.format(iteration,N_fail,i_fail,F_count))
        print('iteration = {}, N_fail = {}, i_fail = {}, F_count = {}, sample size = {}, Pareto points = {}'.format(iteration,N_fail,i_fail,F_count,X.shape[0],count))
        
        if i_fail == 0:
            break
        
    return [N_fail,F_count,direction]
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:31:32 2023

@author: rsdol
"""

# New Successive Pareto Simulation (Prof. Renato Motta - UFPE)
# Test version

import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import matplotlib
import Distribution_class as DC
from scipy import stats

# Pareto front
def domination_filter(X,direction):
    
    start = time.time()
    pdominant = np.arange(X.shape[0])
    next_point_index = 0  # Next index in the pdominant array to search for
    while next_point_index<len(X):
        nondominated_point_mask = np.any(X>X[next_point_index], axis=1) # Search for non-dominated points
        nondominated_point_mask[next_point_index] = True # The point used as reference is non-dominated
        pdominant = pdominant[nondominated_point_mask]  # Remove dominated points
        X = X[nondominated_point_mask] # Remove dominated points
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1 # Next reference point
    end = time.time()
    time1 = end - start
    return pdominant, time1

# Failure direction
def faildir(M,S,fun_G,nRV):
    global transformer
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

def newfaildir(X,pdominant,gd,direction,i_fail):
    
    if i_fail == 0:
        new_direction = direction*(-1) # If failure is not found, search in opposite direction
    elif i_fail >= len(pdominant)*0.7 or i_fail > 10:
        new_direction = direction # If more than 70% of Pareto front is in failure domain, keep direction
    else:
        safe_par = X[pdominant[gd>0]]
        fail_par = X[pdominant[gd<=0]]
        new_direction = np.sign(np.mean(fail_par,axis=0) - np.mean(safe_par,axis=0)) # Change direction following Pareto front

    return new_direction

def x_to_u(X,RV_type,P):
    
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
            X_normal[j,i]=stats.norm.ppf(Px,0,1)
            Seq[j,i]=stats.norm.pdf(X_normal[j,i])/px
            Meq[j,i]=X[j,i]-X_normal[j,i]*Seq[j,i]
    
    return X_normal, Meq, Seq

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

def find_check_points(X,direction):
    
    # Dimension
    n = len(direction)
    
    # Generate n - 1 linearly independent vectors orthogonal to direction
    n = len(direction)
    direction_norm = np.linalg.norm(direction)
    direction_normalized = direction / direction_norm if direction_norm > 1e-10 else direction

    ortho_vectors = []
    while len(ortho_vectors) < n - 1:
        new_vector = np.random.randn(n)
        new_vector -= np.dot(new_vector, direction_normalized) * direction_normalized
        for u in ortho_vectors:
            new_vector -= np.dot(new_vector, u) * u
        norm_new_vector = np.linalg.norm(new_vector)
        if norm_new_vector > 1e-10:
            ortho_vectors.append(new_vector / norm_new_vector)
    
    # Finds the orthonormal basis for the hiper-plane orthogonal to failure direction (Gram Schmidt theorem)
    basis = []
    for v in ortho_vectors:
        for u in basis:
            v -= np.dot(v, u) * u
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-10:
            basis.append(v / norm_v)
    
    # Find projections onto orthogonal basis
    projections = []
    for i in basis:
        projections.append([np.dot(point,i) for point in X])
    projections = np.array(projections)

    # Find projections onto opposite direction
    projection_opposite = []
    for i in np.array([direction/np.linalg.norm(direction)]):
        projection_opposite.append([np.dot(point,i) for point in X])
    projection_opposite = np.array(projection_opposite)
    
    # Find check points (extreme points in each orthogonal direction and opposite direction)
    check_points = []
    for i in range(np.size(projections,axis=0)):
        max_index = np.argmax(projections[i])
        min_index = np.argmin(projections[i])
        extreme_min = projections[i][min_index]*basis[i]
        extreme_max = projections[i][max_index]*basis[i]
        check_points.append((extreme_min))
        check_points.append((extreme_max))
    for i in range(np.size(projection_opposite,axis=0)):
        min_index = np.argmin(projection_opposite[i])
        opposite_min = projection_opposite[i][min_index]*np.array([direction/np.linalg.norm(direction)])[i]
        check_points.append((opposite_min))
    
    return check_points

# New Successive Pareto Simulation (NSPS)
def PSMC(X,fun_G,M,S,RV_type,P,direction=[],width=0,plot=[],max_fail_stack=[]):
    # direction = List with 1 or -1 for each RV (float)
    # width = Width of the vector from mean to safe Pareto points where it is safe to remove points
    #         (It is given in terms of standard deviation, so it is recommended a number between
    #          1.5 and 2 if there are only 2 failure directions (or less) and between 0.1 and 1 for more).
    # plot = Set any value to generate plot figures (PNG) for problems with 2 variables
    # max_fail_stack = Maximum value for the allowed number of iterations without failed Pareto points
    
    # Transform sample X to standard gaussian space
    X, Meq, Seq = x_to_u(X,RV_type,P)
    
    # Initialize variables
    nv = np.size(X,axis=1)
    times = 0; # Count time
    fail_stack = 0 # Stack for how many iterations without failed points are allowed
    if max_fail_stack == []:
        max_fail_stack = 3 # Maximum value for fail stack
    if direction == []:
        direction = faildir(M,S,fun_G,nv); # Find initial failure direction
    else:
        direction = np.array(direction) # Use given initial failure direction
    if width == 0:
        lateral_tol = np.ones(nv); # Width to remove points from sample (related to directions from mean to safe Pareto points)
    else:
        lateral_tol = width*np.ones(nv);
    iteration = 0 # Iteration counter
    F_count = 0 # Function evaluations counter
    N_fail = 0 # Failed points counter
    
    # Check if there is failure in alternative directions (if no width is given)
    # This check is important if the failure function behavior is unknown
    if width == 0:
        check_points = find_check_points(X,direction)
        for i in range(len(check_points)):
            G_check = fun_G(u_to_x(check_points[i],RV_type,P))
            F_count += 1
            if G_check <= 0:
                print('\nNew algorithm on\n')
                width += 1 # If there is one alternative failure direction, active process to maintain some dominated points
                break
        
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
        
        # Find points to keep in the sample
        safe_in = safe_out # safe_in is the set with index of points to maintain in sample
        if not width == 0: # If width = 0, no process is required (all safe dominated points will be removed)
            total_inlimit = set() # Set of points inside width of safe Pareto points (to remove from sample)
            for i in range(len(pdominant)):
                if gd[i]>0: # For safe Pareto points only
                    if width == 1:
                        safe_dist = np.abs(Xaux[pdominant[gd>0]]-Xaux[pdominant[i]]) # Distance between safe Pareto points
                        if np.sum(np.logical_and(np.all(safe_dist <= np.ones(nv),axis=1),np.all(safe_dist > 0.25*np.ones(nv),axis=1))) < 2: # If less than 3 safe Pareto points inside width, do not remove points
                            inlimit_points = set()
                        elif np.sum(np.logical_and(np.all(safe_dist <= 2*np.ones(nv),axis=1),np.all(safe_dist > 0.5*np.ones(nv),axis=1))) < 2:
                            lateral_tol = 0.25*np.ones(nv)
                        elif np.sum(np.logical_and(np.all(safe_dist <= 2*np.ones(nv),axis=1),np.all(safe_dist > np.ones(nv),axis=1))) < 2:
                            lateral_tol = 0.5*np.ones(nv)
                        elif np.sum(np.logical_and(np.all(safe_dist <= 2*np.ones(nv),axis=1),np.all(safe_dist > 1.5*np.ones(nv),axis=1))) < 2:
                            lateral_tol = np.ones(nv)
                        else:
                            lateral_tol = 1.5*np.ones(nv)
                    else:
                        safe_dist = np.abs(Xaux[pdominant[gd>0]]-Xaux[pdominant[i]]) # Distance between safe Pareto points
                        if np.sum(np.logical_and(np.all(safe_dist <= np.ones(nv),axis=1),np.all(safe_dist > 0.25*np.ones(nv),axis=1))) < 2: # If less than 3 safe Pareto points inside width, do not remove points
                            inlimit_points = set()
                        else:
                            lateral_tol = width*np.ones(nv)
                    if np.sum(np.logical_and(np.all(safe_dist <= 2*np.ones(nv),axis=1),np.all(safe_dist > 0.25*np.ones(nv),axis=1))) > 1:
                        #else:
                        # Find perpendicular distance from sample points with respect to vector from mean to safe Pareto point
                        v = np.zeros(nv) - Xaux[pdominant[i]]
                        w = Xaux[pdominant[i]] - Xaux
                        parallel_dist = (np.dot(w, v)/np.dot(v, v))[:,np.newaxis]*v # Calculate the projection of w onto v
                        perpendicular_dist = w - parallel_dist # Calculate the perpendicular vector
                        # Set with points to be removed (inside width limits)
                        inlimit_points = safe_out.intersection(set(n_index[np.all(np.abs(perpendicular_dist)<=lateral_tol,axis=1)])) # Set limit
                        inlimit_points = set(n_index[list(inlimit_points)][np.linalg.norm(parallel_dist[list(inlimit_points)],axis=1) <= np.linalg.norm(v)]) # Remove points until mean point
                else:
                    inlimit_points = set() # If it is not a safe Pareto front, do not remove points
                # Set with points found to remove
                total_inlimit |= inlimit_points
            
            # Update safe_in (points to keep in sample)
            safe_in = (safe_in - set(n_index[pdominant]) - total_inlimit)
            
            # Update safe_out (points to remove from sample)
            safe_out = safe_out - safe_in
        
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
        
        # Update direction for next iteration
        direction = newfaildir(X,pdominant,gd,direction,i_fail)
        
        # Remove from the sample dominant points and safe dominated points
        X = np.delete(X,(list(i_out)),axis=0)
        
        print('iteration = {}, N_fail = {}, i_fail = {}, F_count = {}, sample size = {}, Pareto points = {}'.format(iteration,N_fail,i_fail,F_count,X.shape[0],count))
        
        # Fail stack
        if i_fail == 0:
            fail_stack = fail_stack + 1
            if fail_stack == max_fail_stack:
                break
        else:
            fail_stack = 0
            
    return [N_fail,F_count,direction,times]
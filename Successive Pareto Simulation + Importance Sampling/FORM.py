# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:50:31 2017

@author: Adriano
"""
import scipy as sp
from scipy import stats
import Distribution_class as DC
import numpy as np
#from G_calc import G_calc
#from main_pipe_normas import effective_area as effecArea

########### FORM ############ 
def form(M,RV_type,nRV,P,fun_G,*args, echo_on=True):
    # ponto inicial
    X=np.transpose(M);
    dx = np.max([np.ones(nRV)*1e-4,X*1e-4],axis=0); #diferencas finitas
    Xr=X*0;
    F_count = 0
    
    G=fun_G(X,*args)
    F_count += 1
    #G_step = 1000
    # print('G_0 = '+str(G))
    Seq=np.zeros(nRV)
    Meq=np.zeros(nRV)
    dGdx=np.zeros(nRV)
    ii=0

    #while abs(1-G/G_step)>1e-2 or ii<1e2:
    while abs(G)>1e-5 and ii<50:
    #while abs(G)>1e-6:
        ii=ii+1
        #G_step=G
        # Normal Equivalent Distribution
        for i in range(nRV):
            #print(RV_type[i],X[i],P[i][:])
            Px=DC.cdf(RV_type[i],X[i],P[i][:])
            px=DC.pdf(RV_type[i],X[i],P[i][:])
            if px<1e-25:
                px = 1e-25
            if Px<1e-25:
                Px = 1e-25
            Xr[i]=stats.norm.ppf(Px,0,1)
            Seq[i]=stats.norm.pdf(Xr[i])/px
            Meq[i]=X[i]-Xr[i]*Seq[i]
        
            
        # Comput Gradient
        #[G1,dGdx]=feval(fun_G,X,f_par{:});
        # Finite Difference (dGdx)
        for i in range(nRV):
            X[i] = X[i] + dx[i]
            
            G1=fun_G(X,*args)
            F_count += 1
            
            #print('i '+str(i))
            #print('Xi '+str(X[i]))
            #print(G1+X[4])
            #print(X[3])
            dGdx[i] = (G1 - G)/dx[i]
            X[i]=X[i]-dx[i]
        #print('G1 = '+str(G1))
        
        # Jacobian Transformation, J = sp.diag(Seq);
        dGr = Seq*dGdx
        L = sp.linalg.norm(dGr)
        alpha = dGr/L

        # k = (G-Xr.dot(dGr))/(dGr.dot(dGr));
        # Xr = -k*dGr;

        beta = G/L - Xr.dot(alpha)
        Xr = -beta*alpha
        #print(Xr)
        X = np.transpose(Xr)*Seq+Meq
        
        G=fun_G(X,*args)
        F_count += 1
        if echo_on:
            print('beta = %.6f'%beta+', G = %.10f'%G+', iter = %.0f'%ii)
        #print('Conv G '+str(1-G/G_step))
    MPP = Xr;
    beta=sp.linalg.norm(Xr)
    PF_form=DC.cdf('norm',-beta,[0,1])
    if echo_on:
        print('PF_form = %.19f'%PF_form+'\n')
    # MPP_sobre_M = MPP/M
    # [mdif,km]=[MPP_sobre_M[sp.argmax(MPP_sobre_M)],sp.argmax(MPP_sobre_M)];
    # print('Mdif = %.6f'%mdif+'\n')
    return PF_form, beta, MPP, ii

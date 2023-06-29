# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 22:37:32 2021

@author: rnelli
"""
import pandas as pd
import numpy as np        
def ReLU_stability(n_buses,W,b):    
    Input_NN = pd.read_csv('Data_File/'+str(n_buses)+'/NN_input.csv', header=None)   
    N_hid_l = len(W)-1
    
    N_sample = np.size(Input_NN,0)
    N_dns= np.size(W[0],0)
    Relu=np.zeros((N_sample,N_hid_l,N_dns))
    for i in range(N_sample):
        
        Zk_hat = np.array(b[0]).reshape(1,-1)+np.array(W[0].dot(Input_NN.iloc[i])).reshape(1,-1)
        Zk=np.fmax(Zk_hat,0)   
        Relu[i][0]=Zk> 0
        for k in range(1,N_hid_l):
            Zk_hat = np.array(b[k]).reshape(1,-1)+np.array(W[k].dot(Zk)).reshape(1,-1)
            Zk=np.fmax(Zk_hat,0)   
            Relu[i][k]=Zk> 0 
            
        ReLU_stability_active=np.sum(Relu,0)==N_sample;
        ReLU_stability_inactive=np.sum(Relu,0)==0;

    return ReLU_stability_active, ReLU_stability_inactive
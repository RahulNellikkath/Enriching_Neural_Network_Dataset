import cvxpy as cp
import numpy as np   
import pandas as pd


def MILP_WCG(n_g,n_buses, W,b, Gen_delta,Gen_max):
    
    N_hid_l = len(W)-1
    
    N_lod= np.size(W[0],1)
    N_dns= np.size(W[0],0)
    N_gen= np.size(W[N_hid_l],0)

    # Parameters 

    zk_hat_min =np.ones((N_dns,N_hid_l))*(-10000)
    zk_hat_max =np.ones((N_dns,N_hid_l))*(10000)
    
    u_init = np.ones((N_lod,1))
    l_ini = 0* np.ones((N_lod,1))
       
    zk_hat_max[:,0] = (np.maximum(W[0], 0)@u_init 
                              + np.minimum(W[0], 0)@l_ini ).reshape((N_dns,)) + b[0]
    zk_hat_min[:,0]= (np.minimum(W[0], 0)@u_init 
                              + np.maximum(W[0], 0)@l_ini).reshape((N_dns,)) + b[0]
    for k in range(1,N_hid_l):
        zk_hat_max[:,k] = ((np.maximum(W[k], 0))@np.maximum(zk_hat_max[:,k-1], 0) + 
                              (np.minimum(W[k], 0))@np.maximum(zk_hat_min[:,k-1], 0)).reshape((N_dns,)) + b[k].reshape((N_dns,)) 
        zk_hat_min[:,k] = ((np.minimum(W[k], 0))@np.maximum(zk_hat_max[:,k-1], 0) 
                              + (np.maximum(W[k], 0))@np.maximum(zk_hat_min[:,k-1], 0)).reshape((N_dns,)) + b[k].reshape((N_dns,))
    
    Z_min = cp.Parameter((N_dns,N_hid_l),value=zk_hat_min)
    Z_max = cp.Parameter((N_dns,N_hid_l),value=zk_hat_max)


    # CVXPY Variables
    x = cp.Variable(N_lod)
    Z = cp.Variable((N_dns,N_hid_l),nonneg=True)
    Z_hat = cp.Variable((N_dns,N_hid_l))
    pg_pre = cp.Variable(N_gen)
    ReLU_stat = cp.Variable((N_dns,N_hid_l),boolean=True)

    # Input Domain
    constraints = [ x >= 0 , x <= 1]
    
    ReLU_Stability_check = True
    
    if ReLU_Stability_check == True:
        ReLU_stability_active, ReLU_stability_inactive = ReLU_stability(n_buses,W,b)
    else:
        ReLU_stability_inactive = zk_hat_max<=0
        ReLU_stability_active = zk_hat_min>0
    
    
    for i in range(N_dns):
        for j in range(N_hid_l):
            if ReLU_stability_inactive[i][j] == True:
                constraints += [ReLU_stat[i][j]== False]
            if ReLU_stability_active[i][j] == True:
                constraints += [ReLU_stat[i][j]== True]
                
    # First layer
    constraints += [Z_hat[:,0] - W[0] @ x -b[0] == 0]
    
    # Hidden layers
    for i in range(1,N_hid_l):
        constraints += [Z_hat[:,i] - W[i] @ Z[:,i-1] -b[i] == 0]
    
    #output layers
    constraints += [pg_pre - W[N_hid_l] @ Z[:,N_hid_l-1] -b[N_hid_l] == 0]
    
    # Relu MILP
    constraints += [Z - Z_hat - cp.multiply(ReLU_stat,Z_min) + Z_min <= 0]
    constraints += [Z - Z_hat>= 0]
    constraints += [Z - cp.multiply(ReLU_stat,Z_max) <= 0]       
    
    objective_WC = cp.Minimize(Gen_max[n_g] - Gen_delta[n_g]*pg_pre[n_g])
    
    problem = cp.Problem(objective_WC, constraints)
    problem.solve(solver=cp.GUROBI)

    return max(-objective_WC.value,0)
    

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

import cvxpy as cp
import numpy as np   

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
    
    always_inactive = zk_hat_max<=0
    always_active = zk_hat_min>0
    
    for i in range(N_dns):
        for j in range(N_hid_l):
            if always_inactive[i][j] == True:
                constraints += [ReLU_stat[i][j]== False]
            if always_active[i][j] == True:
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

    Type = 'ReMILP'
    if objective_WC.value < 0:
        
        if Type == 'MILP':
            # Parameters 
            x_0 = cp.Parameter(N_lod, value= x.value)
            WC_violation = objective_WC.value
            # CVXPY Variables
            x_n = cp.Variable(N_lod)
            
            diff_n = cp.Variable(N_lod)
            
            Z = cp.Variable((N_dns,N_hid_l),nonneg=True)
            Z_hat = cp.Variable((N_dns,N_hid_l))
            
            pg_pre_n = cp.Variable(N_gen)
            
            ReLU_stat = cp.Variable((N_dns,N_hid_l),boolean=True)
    
            constraints = [ x_n >= 0 , x_n <= 1]
            
            for i in range(N_dns):
                for j in range(N_hid_l):
                    if always_inactive[i][j] == True:
                        constraints += [ReLU_stat[i][j]==False]
                    if always_active[i][j] == True:
                        constraints += [ReLU_stat[i][j]==True]        
                
        
            # First layer
            constraints += [Z_hat[:,0] - W[0] @ x_n -b[0] == 0]
            
            # Hidden layers
            for i in range(1,N_hid_l):
                constraints += [Z_hat[:,i] - W[i] @ Z[:,i-1] -b[i] == 0]
            
            #output layers
            constraints += [pg_pre_n - W[N_hid_l] @ Z[:,N_hid_l-1] -b[N_hid_l] == 0]
            
            # Relu MILP
            constraints += [Z - Z_hat - cp.multiply(ReLU_stat,Z_min) + Z_min <= 0]
            constraints += [Z - Z_hat>= 0]
            constraints += [Z - cp.multiply(ReLU_stat,Z_max) <= 0]

        if Type == 'ReMILP':
            # Parameters 
            Z_hat_value = Z_hat.value
            x_0 = cp.Parameter(N_lod, value= x.value)
            WC_violation = objective_WC.value
            # CVXPY Variables
            x_n = cp.Variable(N_lod)
            
            diff_n = cp.Variable(N_lod)
            
            Z = cp.Variable((N_dns,N_hid_l),nonneg=True)
            Z_hat = cp.Variable((N_dns,N_hid_l))
            
            pg_pre_n = cp.Variable(N_gen)
            
            ReLU_stat_new = cp.Variable((N_dns,N_hid_l),boolean=True)
    
            constraints = [ x_n >= 0 , x_n <= 1]
            
            for i in range(N_dns):
                for j in range(N_hid_l):
                    if always_inactive[i][j] == True:
                        constraints += [ReLU_stat_new[i][j]==False]
                    if always_active[i][j] == True:
                        constraints += [ReLU_stat_new[i][j]==True]        
                
            
            assumed_constant_active=np.logical_and( Z_hat_value >= 0.2*zk_hat_max, zk_hat_max>=0)
            assumed_constant_inactive=np.logical_and(Z_hat_value <= 0.2*zk_hat_min, zk_hat_min<=0)
            
            for i in range(N_dns):
                for j in range(N_hid_l):
                    if assumed_constant_active[i][j] == True:
                        constraints += [ReLU_stat_new[i][j]==True]
                    elif assumed_constant_inactive[i][j] == True:
                        constraints += [ReLU_stat_new[i][j]==False]                              

            
            # First layer
            constraints += [Z_hat[:,0] - W[0] @ x_n -b[0] == 0]
            
            # Hidden layers
            for i in range(1,N_hid_l):
                constraints += [Z_hat[:,i] - W[i] @ Z[:,i-1] -b[i] == 0]
            
            #output layers
            constraints += [pg_pre_n - W[N_hid_l] @ Z[:,N_hid_l-1] -b[N_hid_l] == 0]
            
            # Relu MILP
            constraints += [Z - Z_hat - cp.multiply(ReLU_stat_new,Z_min) + Z_min <= 0]
            constraints += [Z - Z_hat>= 0]
            constraints += [Z - cp.multiply(ReLU_stat_new,Z_max) <= 0]

        if Type == 'Relax':
            # Parameters 
            x_0 = cp.Parameter(N_lod, value= x.value)
            WC_violation = objective_WC.value
            # CVXPY Variables
            x_n = cp.Variable(N_lod)
            
            diff_n = cp.Variable(N_lod)
            
            Z = cp.Variable((N_dns,N_hid_l),nonneg=True)
            Z_hat = cp.Variable((N_dns,N_hid_l))
            
            pg_pre_n = cp.Variable(N_gen)
            
            
            # Input Domain
            constraints = [ x_n >= 0 , x_n <= 1]

             
            # First layer
            constraints += [Z_hat[:,0] - W[0] @ x_n -b[0] == 0]
            
            # Hidden layers
            for i in range(1,N_hid_l):
                constraints += [Z_hat[:,i] - W[i] @ Z[:,i-1] -b[i] == 0]
            
            #output layers
            constraints += [pg_pre_n - W[N_hid_l] @ Z[:,N_hid_l-1] -b[N_hid_l] == 0]
            
            # Relu MILP
            for i in range(N_dns):
                for j in range(N_hid_l):
                    if always_inactive[i][j] == True:
                        constraints += [Z[i][j] == 0]
                    elif always_active[i][j] == True:
                        constraints += [Z[i][j] - Z_hat[i][j] == 0]
                    else:
                        constraints += [Z[i][j]  - cp.multiply(Z_max[i][j] ,(Z_hat[i][j] -Z_min[i][j] )/(Z_max[i][j] -Z_min[i][j] ))<= 0]
            constraints += [Z - Z_hat>= 0]

            
            
        norm1 = False
        norm_inf = True
        

        if norm1 == True:
            
            ep = cp.Variable(N_lod)
            constraints += [ x_n - x_0 <= ep]
            constraints += [ x_0 - x_n <= ep]
        
            constraints += [Gen_max[n_g] - Gen_delta[n_g]*pg_pre_n[n_g] >= 0]
            
            objective = cp.Minimize(cp.sum(ep))
            
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.GUROBI,reoptimize=True)
            
            if objective.value is not None:   
                ep_value = (objective.value)/N_lod
            else:
                ep_value = 0
                
        elif norm_inf == True:
            
            ep = cp.Variable(1)
            constraints += [ x_n - x_0 <= ep ]
            constraints += [ x_0 - x_n <= ep]
            
            constraints += [Gen_max[n_g] - Gen_delta[n_g]*pg_pre_n[n_g] >= 0.9*WC_violation]
            
            objective = cp.Minimize(ep)
            # objective = cp.Minimize(-1*ep)
            
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.GUROBI,reoptimize=True)
            
            if objective.value is not None:   
                ep_value = (objective.value)
            else:
                ep_value = 0
    
        else :
            constraints += [ x_n - x_0 == diff_n ]
            constraints += [Gen_max[n_g] - Gen_delta[n_g]*pg_pre_n[n_g] >= 0.8*WC_violation]
            objective = cp.Minimize(cp.sum_squares(diff_n))
            
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.GUROBI,reoptimize=True)
            
            if objective.value is not None:   
                ep_value = np.sqrt(objective.value)
            else:
                ep_value = 0
                
        return  max(-objective_WC.value,0) , ep_value,x.value
    return max(-objective_WC.value,0), 0, x.value
    
  
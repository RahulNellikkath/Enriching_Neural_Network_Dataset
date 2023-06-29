# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 16:53:33 2021

@author: rnelli
"""
import torch

import torch.nn as nn

import numpy as np
from PINNs.create_example_parameters import create_example_parameters
from PINNs.create_data import create_data
from PINNs.create_data import create_test_data

from PINNs.MILP_WCG import MILP_WCG
# from multiprocessing import Pool
from Neural_Network.CoreNetwork import NeuralNetwork

from numpy.random import randn, rand

import wandb

def to_np(x):
    return x.detach().numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        training_loss =  100
        
        n_buses=config.test_system 
        simulation_parameters = create_example_parameters(n_buses)
        
        # Getting Training Data
        Dem_train, Gen_train, Volt_train = create_data(simulation_parameters=simulation_parameters)
        
        # Defining the tensors
        Dem_train = torch.tensor(Dem_train).float()
        Gen_train = torch.tensor(Gen_train).float()
        #--------------------------------------------------------------------- 
        # Gen type defines if the data belongs to training dataset or if it 
        # belongs to the additional points collected from the training space
        # type = 1 means it is part of the training set and the generation measurements are present
        #---------------------------------------------------------------------
        Gen_train_type = torch.ones(Gen_train.shape[0],1)
        Volt_train = torch.tensor(Volt_train).float()
        # Volt_train_type = torch.ones(Volt_train.shape[0],1)
        
        num_classes =  Gen_train.shape[1]
        
        Gen_delta=simulation_parameters['true_system']['Gen_delta'] 
        Gen_max=simulation_parameters['true_system']['Gen_max']

        # Test Data
        Dem_test, Gen_test, Volt_test = create_test_data(simulation_parameters=simulation_parameters)
        Dem_test = torch.tensor(Dem_test).float()
        Gen_test = torch.tensor(Gen_test).float()
        Volt_test = torch.tensor(Volt_test).float()

        # NNs for predicting Generation (network_gen) and Volatage (network_Volt)
        network_gen = build_network(Dem_train.shape[1],
                                Gen_train.shape[1],
                                config.hidden_layer_size,
                                config.n_hidden_layers,
                                config.pytorch_init_seed)
        
        network_gen = normalise_network(network_gen, Dem_train, Gen_train)
        
        network_Volt = build_network(Dem_train.shape[1],
                                Volt_train.shape[1],
                                config.hidden_layer_size,
                                config.n_hidden_layers,
                                config.pytorch_init_seed)
        
        network_Volt = normalise_network(network_Volt, Dem_train, Volt_train)
        
        
        Para= list(network_gen.parameters()) + list(network_Volt.parameters())
        
        optimizer = torch.optim.Adam(Para,lr=config.learning_rate)
        lambda1 = lambda epoch: (epoch+1)**(-config.lr_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        
        # The NN will be trained for 200 iterations before enriching the NN database
        # config.epochs should be greater than 200 
        
        for epoch in range(200):
            # Training NN and getting the training loss
            training_loss, Volt_train_loss = train_epoch(network_gen,network_Volt, Dem_train, Gen_train,Volt_train, optimizer,config,simulation_parameters)
            validation_loss = validate_epoch(network_gen, Dem_test,Gen_test)
            scheduler.step()
            wandb.log({"training_loss": training_loss, "validation_loss": validation_loss, "epoch": epoch})

        # initializing the new data set before the enriching begins
        New_Dem_train = Dem_train
        New_Gen_train = Gen_train 
        New_Volt_train = Volt_train
        New_typ  = Gen_train_type
    
        # After the initial training fro 200 epoches the NN data base enriching begins
        for en in range(config.N_enrich):
            
            # Getting new data sets
            Dem_g_violtn,Gen_g_violtn,Volt_g_violtn,typ,X_violations = wc_enriching(network_gen, config, Gen_train, simulation_parameters)
            
            np.savetxt('Test_output/'+str(config.test_system )+'/WC_D_'+'_s' + str(config.pytorch_init_seed) +'_N' + str(config.N_enrich) + '_en' + str(en) +'.csv',X_violations, fmt='%s', delimiter=',')

            # Appending the the existing dataset
            New_Dem_train = torch.cat((New_Dem_train, Dem_g_violtn), 0)
            New_Gen_train = torch.cat((New_Gen_train, Gen_g_violtn), 0)
            New_Volt_train = torch.cat((New_Volt_train, Volt_g_violtn), 0)
            New_typ  = torch.cat((New_typ, typ), 0)
            
            # Shuffling data set before training
            # This will help to have similar data set during batching 
            shuffled_ind=torch.randperm(New_Dem_train.shape[0])
            New_Dem_train = New_Dem_train[shuffled_ind]
            New_Gen_train = New_Gen_train[shuffled_ind]
            New_Volt_train=New_Volt_train[shuffled_ind]
            New_typ = New_typ[shuffled_ind]            
            
            for ep_en in range((config.epochs-200)//(config.N_enrich)):
                #Training with new added collocation points
                training_loss, Volt_train_loss,PF_train_loss = train_GenViolations(network_gen, 
                                                                   network_Volt, 
                                                                   New_Dem_train,
                                                                   New_Gen_train,
                                                                   New_Volt_train, 
                                                                   New_typ,
                                                                   Dem_g_violtn,
                                                                   optimizer,
                                                                   config,
                                                                   simulation_parameters)
                
                epoch+=1
                validation_loss = validate_epoch(network_gen, Dem_test,Gen_test)
                scheduler.step()
                wandb.log({"training_loss": training_loss, "validation_loss": validation_loss,"PF_loss":PF_train_loss, "epoch": epoch})
    
        # Evaluating Worst case violation after NN training
        wandbs=network_gen.state_dict()
        W={}; b={}
        for k in range(config.n_hidden_layers):
            W[k] = wandbs['dense_layers.dense_' +str(k) +'.weight'].data.numpy()
            b[k] = wandbs['dense_layers.dense_' +str(k) +'.bias'].data.numpy()
            
        b[config.n_hidden_layers]=wandbs['dense_layers.output_layer.bias'].data.numpy()
        W[config.n_hidden_layers]=wandbs['dense_layers.output_layer.weight'].data.numpy()
        
        max_wc_g = 0
        max_ep = 0
        for i in range(num_classes):
            wc_g, ep, x_0 = MILP_WCG(i,n_buses,W,b,Gen_delta,Gen_max)
            wandb.log({"WC_G": wc_g ,"Epsilon": ep , "Gen": i})
            if max_wc_g < wc_g:
                max_wc_g = wc_g
                max_ep= ep
        wandb.log({"Max_WC_G": max_wc_g ,"Epsilon_Max_WC": max_ep})
        objective = max_wc_g #/10 +  validation_loss/0.0004
        wandb.log({"objective": objective})
      
def build_network(n_input_neurons, n_output_neurons,hidden_layer_size, n_hidden_layers, pytorch_init_seed):
    model = NeuralNetwork(n_input_neurons, n_output_neurons,
                                         hidden_layer_size=hidden_layer_size,
                                         n_hidden_layers=n_hidden_layers,
                                         pytorch_init_seed=pytorch_init_seed)

    return model.to(device)

def normalise_network(model, Dem_train, Gen_train):
    input_statistics = torch.std_mean(Dem_train, dim=0, unbiased=False)
    output_statistics = torch.std_mean(Gen_train, dim=0, unbiased=False)

    model.normalise_input(input_statistics=input_statistics)
    model.normalise_output(output_statistics=output_statistics)

    return model

def validate_epoch(network_gen, Dem_test,Gen_test):
    criterion = nn.MSELoss()
    output = network_gen.forward(Dem_test)
    validate_loss = criterion(output, Gen_test)
    
    return validate_loss
    

def train_epoch(network_gen,network_Volt, Dem_train, Gen_train,Volt_train, optimizer, config,simulation_parameters):
    n_bus = simulation_parameters['general']['n_buses']
    Gen_delta=simulation_parameters['true_system']['Gen_delta'] 
    Gen_max=simulation_parameters['true_system']['Gen_max']
    Volt_max = simulation_parameters['true_system']['Volt_max']
    Volt_min = simulation_parameters['true_system']['Volt_min']
    # NNs for predicting Generation (network_gen) and Volatage (network_Volt)
    network_gen.train()
    network_Volt.train()
    # initializing parameters
    cur_gen_loss = 0
    cur_volt_loss = 0
    
    # NN loss function
    criterion = nn.MSELoss()
    
    # Defining Data set slicing algorithm
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    
    
    num_samples_train = Dem_train.shape[0]
    num_batches_train = int(num_samples_train // config.batch_size)
    
    RELU = nn.ReLU()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, config.batch_size)
        
        # NN_Gen
        Gen_output = network_gen.forward(Dem_train[slce])
        Gen_target = Gen_train[slce]
        # computeing NN prediction error
        batch_loss_gen = criterion(Gen_output, Gen_target)
        
        # Computing Gen Violations
        batch_loss_gen_violation= config.GenV_weight*torch.mean(torch.square(RELU(Gen_output*torch.from_numpy(np.transpose(Gen_delta)) - torch.from_numpy(np.transpose(Gen_max)))))
        batch_loss= batch_loss_gen +  batch_loss_gen_violation     
        
        # NN_Volt
        Volt_output = network_Volt.forward(Dem_train[slce])
        Volt_target = Volt_train[slce]
        batch_loss_volt = criterion(Volt_output, Volt_target)
        Volt_sqre=Volt_output[:,0:n_bus]**2 + Volt_output[:,n_bus:2*n_bus]**2
        batch_loss_volt_violation=config.GenV_weight*torch.mean(torch.square(RELU(Volt_sqre - Volt_max**2))+torch.square(RELU(Volt_sqre - Volt_min**2)))
        batch_loss += batch_loss_volt + batch_loss_volt_violation
        
        # Calculating Power Flow error
        batch_PF_loss = power_flow_check(Dem_train[slce], Gen_output, Volt_output, simulation_parameters)
        batch_loss += config.PF_weight*batch_PF_loss
        
        # Computing batch loss gradient
        batch_loss.backward()
        optimizer.step()
        
        # Storing for looging to WandB
        cur_gen_loss += batch_loss_gen 
        cur_volt_loss += batch_loss_volt
        
    Gen_train_loss = (cur_gen_loss / config.batch_size)
    Volt_train_loss = (cur_volt_loss / config.batch_size)
    
    return Gen_train_loss, Volt_train_loss

def power_flow_check(Load, Gen, Volt, simulation_parameters):
    n_bus = simulation_parameters['general']['n_buses']
    Y = torch.tensor(simulation_parameters['true_system']['Y'].astype(np.float32))
    Ybr = torch.tensor(simulation_parameters['true_system']['Ybr'].astype(np.float32))
    # Incidence matrix
    IM = torch.tensor(simulation_parameters['true_system']['IM'].astype(np.float32))
    
    g_bus = simulation_parameters['true_system']['g_bus']
    # Maping generators to bus
    Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'].astype(np.float32))
    # maping loads to buses
    Map_L = torch.tensor(simulation_parameters['true_system']['Map_L'].astype(np.float32))    
    Gen_delta=torch.tensor(np.transpose(simulation_parameters['true_system']['Gen_delta']).astype(np.float32)) 
    Dem_max=torch.tensor(np.transpose(simulation_parameters['true_system']['Dem_max']).astype(np.float32))
    
    S_base=100
    
    # Line limits
    L_limit = torch.tensor(simulation_parameters['true_system']['L_limit'].astype(np.float32))
    n_line = simulation_parameters['true_system']['n_line']
    
    # PowerFlow Equation
    PF_error = torch.mean(torch.abs(Volt[:,n_bus + g_bus[0]-1]))


    RELU = nn.ReLU()
    # Computing Line Flow Violation
    Ibr = Volt@torch.transpose(Ybr@(IM),0,1) # I =V*Y or V/Z
    PF_error += torch.mean(RELU(torch.sqrt(torch.square(Ibr[:,0:n_line])+torch.square(Ibr[:,n_line:2*n_line]))- L_limit))

    Iinj = Volt@torch.transpose(Y,0,1) # I =V*Y
    # finding the atcual values from the normalized input
    Load_act = (Load*Dem_max*0.4 + Dem_max*0.6)@Map_L
    Gen_delta_map = Gen_delta@Map_g
    # finding the atcual values from the normalized output values
    Gen_act = (Gen*Gen_delta)@Map_g
    for i in range(n_bus):
        i_nb = i+n_bus
        P_inj = Iinj[:,i:i+1]*Volt[:,i:i+1] + Iinj[:,i_nb:i_nb+1]*Volt[:,i_nb:i_nb+1]
        Q_inj = Iinj[:,i:i+1]*Volt[:,i_nb:i_nb+1] - Iinj[:,i_nb:i_nb+1]*Volt[:,i:i+1]
        
        PF_error += torch.mean(torch.abs(P_inj*S_base + Load_act[:,i:i+1] - Gen_act[:,i:i+1])/(Gen_delta_map[:,i]+1))
        PF_error += torch.mean(torch.abs(Q_inj*S_base + Load_act[:,i_nb:i_nb+1] - Gen_act[:,i_nb:i_nb+1])/(Gen_delta_map[:,i_nb]+1))
    
    return PF_error

def train_GenViolations(network_gen,network_Volt, Dem_train, Gen_train,Volt_train, typ, Dem_g_violtn, optimizer, config,simulation_parameters):
    n_bus = simulation_parameters['general']['n_buses']
    Gen_delta=simulation_parameters['true_system']['Gen_delta'] 
    Gen_max=simulation_parameters['true_system']['Gen_max']
    Volt_max = simulation_parameters['true_system']['Volt_max']

    # NN
    network_gen.train()
    network_Volt.train()
    cur_gen_loss = 0
    cur_volt_loss = 0
    cur_PF_loss =0
    criterion = nn.MSELoss()
    
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    
    num_samples_train = Dem_train.shape[0]
    num_batches_train = int(num_samples_train // config.batch_size)
    RELU = nn.ReLU()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, config.batch_size)
        
        # Evaluating NN_Gen
        Gen_output = network_gen.forward(Dem_train[slce])
        Gen_target = Gen_train[slce]
        batch_loss_Gen= criterion(Gen_output*typ[slce], Gen_target*typ[slce])
        batch_loss_gen_violation = config.GenV_weight*torch.mean(RELU(Gen_output*torch.from_numpy(np.transpose(Gen_delta)) - torch.from_numpy(np.transpose(Gen_max))))
        batch_loss = batch_loss_Gen + batch_loss_gen_violation
        
        # Evaluating NN_Voltage
        Volt_output = network_Volt.forward(Dem_train[slce])
        Volt_target = Volt_train[slce]
        batch_loss_volt = criterion(Volt_output*typ[slce], Volt_target*typ[slce])
        Volt_sqre =Volt_output[:,0:n_bus]**2 + Volt_output[:,n_bus:2*n_bus]**2
        batch_loss_volt_violation = config.GenV_weight*torch.mean(torch.square(RELU(Volt_sqre - Volt_max**2)))
        batch_loss += batch_loss_volt + batch_loss_volt_violation
        
        # Evaluating powerflow
        batch_PF_loss = power_flow_check(Dem_train[slce], Gen_output, Volt_output, simulation_parameters)

        batch_loss += config.PF_weight*batch_PF_loss

        # compute gradients given loss
        batch_loss.backward()
        optimizer.step()
        cur_gen_loss += batch_loss_Gen 
        cur_volt_loss += batch_loss_volt
        cur_PF_loss += batch_PF_loss
    
    Gen_output = network_gen.forward(Dem_g_violtn)
    Volt_output = network_Volt.forward(Dem_g_violtn)
    Volt_sqre =Volt_output[:,0:n_bus]**2 + Volt_output[:,n_bus:2*n_bus]**2
    Gen_train_loss = (cur_gen_loss / config.batch_size)
    Volt_train_loss = (cur_volt_loss / config.batch_size)
    PF_train_loss = (cur_PF_loss / config.batch_size)    
    return Gen_train_loss, Volt_train_loss,PF_train_loss
    
def wc_enriching(network_gen, config, Gen_train,simulation_parameters):
    num_classes =  Gen_train.shape[1]
    n_buses=config.test_system
    n_points=config.n_points
    Gen_delta=simulation_parameters['true_system']['Gen_delta'] 
    Gen_max=simulation_parameters['true_system']['Gen_max']
    n_lbus = simulation_parameters['true_system']['n_lbus']
    
    # Getting NN Weights and Biases
    wandbs=network_gen.state_dict()
    W={}; b={}
    for k in range(config.n_hidden_layers):
        W[k] = wandbs['dense_layers.dense_' +str(k) +'.weight'].data.numpy()
        b[k] = wandbs['dense_layers.dense_' +str(k) +'.bias'].data.numpy()
        
    b[config.n_hidden_layers]=wandbs['dense_layers.output_layer.bias'].data.numpy()
    W[config.n_hidden_layers]=wandbs['dense_layers.output_layer.weight'].data.numpy()
    
    N_lod= np.size(W[0],1)
    
    # initializing datasets
    # x_g_violation gives the load values caused the worst-case generation constraint violation
    x_g_violations=np.zeros((1,N_lod))
    X_violations= np.array([]).reshape(0,2*n_lbus)
    
    max_wc_g = 0
    wc_gs= {}
    eps= {}
    x_0s = {}
    for i in range(int(num_classes)):
        # print(i)
        wc_g,ep,x_0 = MILP_WCG(i,n_buses,W,b,Gen_delta,Gen_max)
        wc_gs[i] = max(wc_g,0)
        eps[i]=ep
        x_0s[i]=x_0
        if max_wc_g < wc_g:
            max_wc_g = wc_g
        X_violations= np.concatenate((X_violations,x_0s[i] .reshape(1,2*n_lbus)), axis=0) 
        
    total_wc_gs = sum(wc_gs.values())
    if (total_wc_gs > 0):
        for i in range(int(num_classes)): 
            if int(np.ceil(n_points*wc_gs[i]/total_wc_gs))> 1:
                x_g_violation = (eps[i]*config.std_ep) * randn(int(np.ceil(n_points*wc_gs[i]/total_wc_gs)),N_lod) + x_0s[i]
                x_g_violations = np.concatenate((x_g_violations,x_g_violation), axis=0)
    
    if n_points == 0:
        x_g_violations =  rand(int(2000),N_lod)
    
    x_g_vio_ts = torch.tensor(x_g_violations).float()
    y_g_vio_ts = torch.zeros(x_g_vio_ts.shape[0], Gen_train.shape[1])
    y_v_vio_ts = torch.zeros(x_g_vio_ts.shape[0], 2*n_buses)
    y_g_type = torch.zeros(x_g_vio_ts.shape[0], 1)
    
    wandb.log({"Max_WC_G_iter": max_wc_g})
    
    return x_g_vio_ts,y_g_vio_ts,y_v_vio_ts,y_g_type,X_violations
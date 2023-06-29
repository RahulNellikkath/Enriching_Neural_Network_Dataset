import numpy as np
import pandas as pd

def create_data(simulation_parameters):
    
    n_buses=simulation_parameters['general']['n_buses'] 
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    s_point = simulation_parameters['data_creation']['s_point']
    
    L_Val=pd.read_csv('Data_File/'+str(n_buses)+'/NN_input.csv').to_numpy()[s_point:s_point+n_data_points][:] 
    Gen_out = pd.read_csv('Data_File/'+str(n_buses)+'/NN_output_PQ.csv').to_numpy()[s_point:s_point+n_data_points][:]
    V_out = pd.read_csv('Data_File/'+str(n_buses)+'/NN_output_V.csv').to_numpy()[s_point:s_point+n_data_points][:]
    
    x_training = L_Val
    return x_training, Gen_out, V_out


def create_test_data(simulation_parameters):

    n_buses=simulation_parameters['general']['n_buses'] 
    n_test_data_points = simulation_parameters['data_creation']['n_test_data_points']
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    s_point = simulation_parameters['data_creation']['s_point']
    n_total = n_data_points + n_test_data_points

    
    L_Val=pd.read_csv('Data_File/'+str(n_buses)+'/NN_input.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
    Gen_out = pd.read_csv('Data_File/'+str(n_buses)+'/NN_output_PQ.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
    V_out = pd.read_csv('Data_File/'+str(n_buses)+'/NN_output_V.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
    # y_gen_data = np.array(Gen_out) 

    
    x_test = np.concatenate([L_Val], axis=0)
    return x_test, Gen_out, V_out
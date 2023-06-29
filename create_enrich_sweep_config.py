import os
import itertools

from types import SimpleNamespace

import wandb
from multiprocessing import Pool


import NN_Enrich_Training

def create_enrich_sweep_config():
    sweep_config = {'program': 'NN_Enrich_Training.py',
                    'method': 'random',
                    'project' : "WC_Enrich",
                    'entity' : "rnelli"}

    metric = {
        'name': 'max_wc_g',
        'goal': 'minimize'
    }

    parameters_dict = {
        'test_system':{'value': 39},
        'hidden_layer_size': {'value': 20},
        'n_hidden_layers': {'value': 3},
        'epochs': {'value': 600},
        'batch_size': {'value': 1000},
        'learning_rate': {'value': 0.01},
        'lr_decay': {'value': 0.9},
        'dataset_split_seed': {'value': 0},
        'pytorch_init_seed': {'value': 2},
        'GenV_weight': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.001,
            'max': 100
          },
        'PF_weight': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.001,
            'max': 100
          },
        'N_enrich' : {'value': 2},
        'n_points': {'value': 0},
        'std_ep' :  {'value': 1/3},
    }


    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric

    return sweep_config

# def create_enrich_sweep_config():
#     sweep_config = {'program': 'NN_Training.py',
#                     'method': 'grid',
#                     'project' : "WC_Enrich",
#                     'entity' : "rnelli"}

#     metric = {
#         'name': 'Max_WC_G',
#         'goal': 'minimize'
#     }

#     parameters_dict = {
#         'test_system':{'value': 118},
#         'hidden_layer_size': {'value': 20},
#         'n_hidden_layers': {'value': 3},
#         'epochs': {'value': 1000},
#         'batch_size': {'value': 1000},
#         'learning_rate': {'value': 0.01},
#         'lr_decay': {'value': 0.95},
#         'pytorch_init_seed': {'values': [10,11,12,13,14]},
#         'dataset_split_seed': {'value': 1},
#         'GenV_weight': {'value': 0.5},
#         'N_enrich' : {'value': 2},
#         'std_ep' :  {'value': 1/3},
#     }

#     sweep_config['parameters'] = parameters_dict
#     sweep_config['metric'] = metric

#     return sweep_config

def setup_sweep():
    wandb.login()
    sweep_config  = create_enrich_sweep_config()
    
    sweep_id = wandb.sweep(sweep_config, project="WC_Enrich_118bus")
    return sweep_id

if __name__ == '__main__':
    
    sweep_id = setup_sweep() 
    wandb.agent(sweep_id, NN_Enrich_Training.train, count=20)
    

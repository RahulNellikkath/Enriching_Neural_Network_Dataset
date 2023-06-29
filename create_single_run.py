from types import SimpleNamespace

import wandb

import NN_Enrich_Training


def create_config():
    parameters_dict = {
        'test_system':39,
        'hidden_layer_size': 20,
        'n_hidden_layers': 3,
        'epochs': 1000,
        'batch_size': 1000,
        'learning_rate': 0.01,
        'lr_decay': 0.97,
        'dataset_split_seed': 10,
        'pytorch_init_seed': 1,
        'GenV_weight': 1,
        'PF_weight': 0.1,
        'N_enrich': 2,
        'n_points':100,
        'std_ep' : 1/4,
    }
    config = SimpleNamespace(**parameters_dict)
    return config


def train_single_run():
    wandb.login()
    run_config = create_config()
    NN_Enrich_Training.train(config=run_config)
    # return sweep_id


if __name__ == '__main__':
    train_single_run()

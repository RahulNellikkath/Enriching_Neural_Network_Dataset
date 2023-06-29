from types import SimpleNamespace

import wandb

import NN_Enrich_Training

if __name__ == '__main__':
    
    sweep_id = "rnelli/WC_Enrich_57bus/a14y7ehe"
    
    wandb.agent(sweep_id, NN_Enrich_Training.train, count=20)

    
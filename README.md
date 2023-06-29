# Enriching_Neural_Network_Dataset

This repository contains the code for Physics-Informed Neural Network for AC Optimal Power Flow applications and the worst case guarantees
When publishing results based on this data/code, please cite: R. Nellikkath and S. Chatzivasileiadis "Physics-Informed Neural Networks for AC Optimal Power Flow", 2021. Available online: https://doi.org/10.48550/arXiv.2303.13228

Author: Rahul Nellikkath E-mail: rnelli@elektro.dtu.dk

This code is distributed WITHOUT ANY WARRANTY, without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

This code requires the following:

conda(python) and Pytorch installations with environment activated
CVXPY 
Gurobi (Gurobi V9.1.2, https://www.gurobi.com/)

The data for the test cases are reproduced from the IEEE PES Power Grid Library - Optimal Power Flow - v19.05 (https://github.com/power-grid-lib/pglib-opf)

To re-create the simulation results run create_enrich_sweep_config.py to start the WANDB sweep. This will build the proposed NN for 39 bus system.

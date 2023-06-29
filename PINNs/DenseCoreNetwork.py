# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:12:38 2021

@author: Rahul N
"""
import torch
#torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.parameter import Parameter

class Net(nn.Module):

    def __init__(self, num_features, num_hidden1,num_hidden2, num_hidden3, num_output,dropout):
        super(Net, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden1, num_features)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden1), 0))
        # hidden layer 1
        self.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden2, num_hidden1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_hidden2), 0))
        # hidden layer 2
        self.W_3 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden3, num_hidden2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_hidden3), 0))
        # hidden layer 3
        self.W_4 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden3)))
        self.b_4 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        # define activation function in constructor
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        #Layer 1
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        # x = self.dropout(x)
        #Layer 2
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        
        x = F.linear(x, self.W_3, self.b_3)
        x = self.activation(x)
        # x = self.dropout(x)
        #Layer 3
        x = F.linear(x, self.W_4, self.b_4)
        return x




class Net_seed(nn.Module):

    def __init__(self, num_features, num_hidden1,num_hidden2, num_hidden3, num_output,dropout,seed):
        torch.manual_seed(seed)    
        super(Net_seed, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden1, num_features)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden1), 0))
        # hidden layer 1
        self.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden2, num_hidden1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_hidden2), 0))
        # hidden layer 2
        self.W_3 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden3, num_hidden2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_hidden3), 0))
        # hidden layer 3
        self.W_4 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden3)))
        self.b_4 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        # define activation function in constructor
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        #Layer 1
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        # x = self.dropout(x)
        #Layer 2
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        
        x = F.linear(x, self.W_3, self.b_3)
        x = self.activation(x)
        # x = self.dropout(x)
        #Layer 3
        x = F.linear(x, self.W_4, self.b_4)
        return x

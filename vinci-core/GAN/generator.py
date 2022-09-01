# Generator class

# Author E.Omer Gul

import torch
from torch import nn

class Generator(nn.Module):

    def __init__(self):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,16), # two inputs first layer 16 Ns
            nn.ReLU(), #ReLU activation function for Ns

            nn.Linear(16,32), # second layer 32 Ns
            nn.ReLU(),

            nn.Linear(32,2), # two outputs 
        )

    def forward(self,x): # x is input, no processing
        return self.model(x)

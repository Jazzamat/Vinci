# Discriminator class

# Author E.Omer Gul

import torch
from torch import nn

class Discriminator(nn.Module):
    
    def __init__(self):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,256), # input is 2D and the first hidden layer has 256 Ns (neurons)
            nn.ReLU(), # the first layer activation is ReLU (Rectified Linear Unit) function
            nn.Dropout(0.3), #to avoid overfitting

            nn.Linear(256,128), # second layer 128 Ns
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64), # thrid layer 64 Ns
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64,1), # 1 output
            nn.Sigmoid(), #The output is composed of a single neuron with sigmoidal activation to represent a probability.
        )

    def forward(self,x): # x represents th input (2D tensor). Just raw data input without any preprocessing
        return self.model(x)
         
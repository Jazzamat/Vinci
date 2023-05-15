

from email.mime import base
from hashlib import new
from base import BaseVAE
from torch import nn
class VAE:

    def __init__(self,in_channels,latent_dim,hidden_dims=None):
        super(VAE, self).__init__()


        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [64,128,256,512]

        #=================== ENCODER =====================

        modules = []
        for hidden_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,out_channels=hidden_dim,kernel_size=3,stride=2,padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU()
                )
            )
               
            in_channels = hidden_dim
            
   
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        #=================== DECODER =====================

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],hidden_dims[i+1],kernel_size=3,stride=2,padding=1,output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],hidden_dims[-1],kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),  
            nn.Tanh()     
        )



    def forward(self):
        raise NotImplementedError




if __name__ == "__main__":
    vae =  VAE(3,5)

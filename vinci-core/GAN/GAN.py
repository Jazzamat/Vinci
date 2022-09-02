from re import L
import torch
from torch import nn
from time import sleep

import math
import matplotlib.pyplot as plt
from discriminator import Discriminator
from generator import Generator

RANDOM_SEED = 111

# training parameters
LR = 0.001 # learning rate, (step size in the gradient decent)
NUM_EPOCHS = 300
LOSS_FUNCTION = nn.BCELoss() # loss = error. Binary Cross Entropy function
OPTIMIZER = torch.optim.Adam


# sleep(0.5) 
if (torch.cuda.is_available()):
    print("GPU driver and CUDA is enabled")
else:
    print("GPU driver and CUDA is NOT enabled")

# sleep(0.5) 
print(f"Random seed set to {RANDOM_SEED}")
torch.manual_seed(RANDOM_SEED)

# sleep(0.5) 
print(f"Creating training data...")
train_data_length = 1024
train_data = torch.zeros((train_data_length,2))
train_data[:,0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:,1] = torch.sin(train_data[:,0])
train_labels = torch.zeros(train_data_length) # unused but need for pytorch
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]

# sleep(0.5) 
print(f"Training data OK")


# sleep(0.5) 
print(f"Plotting...")

x = train_data[:, 0]
y = train_data[:, 1]
plt.plot(x,y,'.')
# plt.show()

# sleep(0.5) 
print(f"Creating PyTorch data loader...")
batch_size = 32
train_loader =  torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
# sleep(0.5) 
print(f"Data loader OK")


# sleep(0.5) 
print(f"Creating Discriminator instance")
discriminator = Discriminator()

# sleep(0.5) 
print(f"Discriminator instance OK")



# sleep(0.5) 
print(f"Creating Generator instance")
generator = Generator()

# sleep(0.5) 
print(f"Generator instance OK")


# sleep(0.5) 
print(f"\n --- INITIATING TRAINING --- \n")

# sleep(0.5) 
print(f"Paramaters:")
print(f"    Learning rate: {LR}")
print(f"    Number of epochs: {NUM_EPOCHS}")
print(f"    Loss function: {LOSS_FUNCTION}")

# sleep(0.5) 
print(f"Setting optimizers:")
# PyTorch implements various weight update rules for model training in torch.optim. 
# We use the Adam algorithm to train the discriminator and generator models. 
optimizer_discriminator = OPTIMIZER(discriminator.parameters(),lr=LR)
optimizer_generator = OPTIMIZER(generator.parameters(),lr=LR)
# sleep(0.5) 
print(f"Optimizers OK")


sleep(0.5) 
print(f"Training...")
sleep(1) 
# training loop
for epoch in range(NUM_EPOCHS):
    # print(epoch)
    for n, (real_samples,_) in enumerate(train_loader):

        # data for training the discriminator
        real_samples_labels = torch.ones((batch_size,1))
        latent_space_samples = torch.randn((batch_size,2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size,1))
        all_samples = torch.cat((real_samples,generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator

        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = LOSS_FUNCTION(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()


        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 2))

        # Training the generator

        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = LOSS_FUNCTION(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()


        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

        



generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
plt.show()

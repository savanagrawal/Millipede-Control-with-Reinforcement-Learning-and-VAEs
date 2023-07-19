'''
The custom VAE training is a part of the project, 'Millipede-Control-with-Reinforcement-Learning-and-VAEs'. This script demonstrates 
the implementation of a VAE training for taking out the latency space from observation space of trained ant. The VAE algorithm in this
script is made with the help of tutorial provided online [1]. This script also shows latency plot in 2 dimesnion and prints decoded data.

[1] https://avandekleut.github.io/vae/

---------------------------------
@author: Savan Agrawal
@file: vae_train_one.py
@version: 0.1
---------------------------------
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LabeledDataset(torch.utils.data.Dataset):
    """
    A class used to represent labeled dataset
    """
    def __init__(self, observations, labels):
        super(LabeledDataset, self).__init__()
        self.observations = observations
        self.labels = labels

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]
    
class VariationalEncoder(nn.Module):
    """
    A class used to represent the Variational Encoder module of the VAE
    """
    def __init__(self, input_dims, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dims, 16)
        self.linear2 = nn.Linear(16, latent_dims)
        self.linear3 = nn.Linear(16, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class Decoder(nn.Module):
    """
    A class used to represent the Decoder module of the VAE
    """
    def __init__(self, input_dims, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 16)
        self.linear2 = nn.Linear(16, input_dims)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 27))
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dims, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dims, latent_dims)
        self.decoder = Decoder(input_dims, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
class VariationalAutoencoderTraining:
    """
    Class to handle training and testing of a Variational Autoencoder (VAE).
    """
    
    def __init__(self, device, filepath, dataset_path, batch_size=32):
        """
        Initialize training configurations.
        """
        torch.manual_seed(0)
        plt.rcParams['figure.dpi'] = 200

        self.device = device
        self.filepath = filepath
        self.batch_size = batch_size
        self.observations = None
        self.labels = None
        self.dataloader = None
        self.dataset_path = dataset_path

    def load_data(self):
        """
        Load and preprocess data.
        """
        # Load raw data
        observations = np.load(self.dataset_path + 'observations10k.npy')
        actions = np.load(self.dataset_path + 'actions10k.npy')
        rewards = np.load(self.dataset_path + 'rewards10k.npy')

        # Prepare the data
        self.observations = torch.tensor(observations).float()
        actions = torch.tensor(actions).float()
        rewards = torch.tensor(rewards).float()

        # Assign labels based on the order of the observations
        labels = np.zeros(len(self.observations))
        labels[10001:20002] = 1
        labels[20002:30003] = 2
        labels[30003:] = 3

        self.labels = torch.tensor(labels).long()

        # Create a LabeledDataset
        dataset = LabeledDataset(self.observations, self.labels)

        # Use the LabeledDataset with a DataLoader
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, vae, epochs=200):
        """
        Train the VAE.
        """
        opt = torch.optim.Adam(vae.parameters())
        for epoch in range(epochs):
            for x, y in self.dataloader:
                x = x.to(self.device) # GPU
                opt.zero_grad()
                x_hat = vae(x)
                loss = ((x - x_hat)**2).sum() + vae.encoder.kl
                loss.backward()
                opt.step()
        return vae

    def save_model(self, vae, model_path):
        """
        Save the trained VAE model.
        """
        torch.save(vae.state_dict(), model_path)

    def load_model(self, vae, model_path):
        """
        Load a trained VAE model.
        """
        vae.load_state_dict(torch.load(model_path, map_location=self.device))
        return vae

    def plot_latent(self, autoencoder, num_batches=100):
        """
        Plot the latent space.
        """
        for i, (x, y) in enumerate(self.dataloader):
            z = autoencoder.encoder(x.to(self.device))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
            if i > num_batches:
                plt.colorbar()
                break
        plt.show()

    def print_reconstructed(self, autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
        """
        Plot the reconstructed space.
        """
        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                z = torch.Tensor([[x, y]]).to(self.device)
                x_hat = autoencoder.decoder(z)
                print(x_hat.shape)

def main():
    # Create a VariationalAutoencoderTraining instance
    vae_training = VariationalAutoencoderTraining(device='cpu', filepath='./src/vae-training/assets/',
                                                  dataset_path='./src/dataset-creator/default-ant/assets/')

    # Load data
    vae_training.load_data()

    # Initialize the model
    num_features = vae_training.observations.shape[-1] * vae_training.observations.shape[-2]
    latent_dims = 2
    vae = VariationalAutoencoder(num_features, latent_dims)

    # Train the model
    # vae = vae_training.train(vae, epochs=200)

    # # Save the model
    # vae_training.save_model(vae, './src/vae-training/assets/vae-train-one.pth')

    # Load the model
    vae = vae_training.load_model(vae, './src/vae-training/assets/vae-train-one.pth')

    # Plot latent space
    vae_training.plot_latent(vae)

    # Plot reconstructed space
    vae_training.print_reconstructed(vae)

if __name__ == '__main__':
    main()
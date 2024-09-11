import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class VAE_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]

class VAE(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, encoding_size),
            nn.LeakyReLU(0.2))
            
        self.mean_layer = nn.Linear(encoding_size, 2)
        self.var_layer = nn.Linear(encoding_size, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, encoding_size),
            nn.LeakyReLU(0.2),
            nn.Linear(encoding_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, input_size),
            nn.Sigmoid())
    
    def encode(self, x):
        x = self.encoder(x)
        mean, var = self.mean_layer(x), self.var_layer(x)
        return mean, var
      
    def decode(self, x):
        return self.decoder(x)
            
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var*epsilon
        return z

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.reparameterization(mean, var)
        x = self.decode(z)
        return x, mean, var
    
# Create the vae
def vae_create(input_size, encoding_size):
  input_size = int(input_size)
  encoding_size = int(encoding_size)
  
  vae = VAE(input_size, encoding_size)
  vae = vae.float()
  return vae  

# Define specific VAE Loss Function
def loss_function(outputs, inputs, mean, var):
    reproduction_loss = nn.functional.binary_cross_entropy(outputs, inputs, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ var - mean.pow(2) - var.exp())

    return reproduction_loss + KLD

# Train the vae
def vae_train(vae, train_loader, num_epochs = 1000, learning_rate = 0.001):
  optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

  for epoch in range(num_epochs):
      running_loss = 0.0
      for data in train_loader:
          inputs, _ = data
          inputs = inputs.float()
          inputs = inputs.view(inputs.size(0), -1)
          optimizer.zero_grad()
          outputs, mean, var = vae(inputs)
          loss = loss_function(outputs, inputs, mean, var)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
#      if (epoch + 1) % 100 == 0:
#          print('Epoch {} Loss: {:.4f}'.format(epoch+1, running_loss/len(train_loader)))

  return vae

def vae_fit(vae, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)
  
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = VAE_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  vae = vae_train(vae, train_loader, num_epochs = 1000, learning_rate = 0.001)
  
  return vae



def vae_encode_data(vae, data_loader):
  # Encode the synthetic time series data using the trained vae
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      inputs = inputs.view(inputs.size(0), -1)
      encoded = vae.mean_layer(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def var_autoencoder_encode(vae, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = VAE_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = vae_encode_data(vae, train_loader)
  
  return(encoded_data)


def var_autoencoder_encode_decode_data(vae, data_loader):
  # Encode the synthetic time series data using the trained vae
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      inputs = inputs.view(inputs.size(0), -1)
      mean, var = vae.encode(inputs)
      z = vae.reparameterization(mean, var)
      decoded = vae.decode(z)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def var_autoencoder_encode_decode(vae, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = VAE_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = var_autoencoder_encode_decode_data(vae, train_loader)
  
  return(encoded_decoded_data)
  

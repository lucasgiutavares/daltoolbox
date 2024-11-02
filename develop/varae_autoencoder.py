import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from random import sample
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
    def __init__(self, input_size, encoding_size, mean_var_size=6):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, encoding_size),
            nn.LeakyReLU(0.2))
            
        self.mean_layer = nn.Linear(encoding_size, mean_var_size)
        self.var_layer = nn.Linear(encoding_size, mean_var_size)

        self.decoder = nn.Sequential(
            nn.Linear(mean_var_size, encoding_size),
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
def vae_create(input_size, encoding_size, mean_var_size):
  input_size = int(input_size)
  encoding_size = int(encoding_size)
  mean_var_size = int(mean_var_size)
  
  vae = VAE(input_size, encoding_size, mean_var_size)
  vae = vae.float()
  return vae  

# Define specific VAE Loss Function
def criterion(outputs, inputs, mean, var):
    reproduction_loss = nn.functional.binary_cross_entropy(outputs, inputs, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ var - mean.pow(2) - var.exp())

    return reproduction_loss + KLD

# Train the vae
def vae_train(vae, train_loader, val_loader, num_epochs = 1000, learning_rate = 0.001, return_loss=False):
  optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

  train_loss = []
  val_loss = []
  
  for epoch in range(num_epochs):
      train_epoch_loss = []
      val_epoch_loss = []
      # Train
      vae.train()
      for train_data in train_loader:
          train_input, _ = train_data
          train_input = train_input.float()
          optimizer.zero_grad()
          train_output, train_mean, train_var = vae(train_input)
          train_batch_loss = criterion(train_output, train_input, train_mean, train_var)
          train_batch_loss.backward()
          optimizer.step()
          train_epoch_loss.append(train_batch_loss.item())
          
          
      # Validation
      vae.eval()
      for val_data in val_loader:
          val_input, _ = val_data
          val_input = val_input.float()
          val_output = vae(val_input)
          val_output, val_mean, val_var = vae(val_input)
          val_batch_loss = criterion(val_output, val_input, val_mean, val_var)
          val_epoch_loss.append(val_batch_loss.item())
          
      train_loss.append(np.mean(train_epoch_loss))
      val_loss.append(np.mean(val_epoch_loss))

  if return_loss:
    return vae, train_loss, val_loss
  else:
    return vae

def vae_fit(vae, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001, return_loss=False):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)
  
  array = data.to_numpy()
  array = array[:, :]
  
  val_sample = sample(range(1, array.shape[0], 1), k=int(array.shape[0]*0.3))
  train_sample = [v for v in range(1, array.shape[0], 1) if v not in val_sample]
  
  train_data = array[train_sample, :]
  val_data = array[val_sample, :]
  
  ds_train = VAE_TS(train_data)
  ds_val = VAE_TS(val_data)
  train_loader = DataLoader(ds_train, batch_size=batch_size)
  val_loader = DataLoader(ds_val, batch_size=batch_size)
  
  if return_loss:
    vae, train_loss, val_loss = vae_train(vae, train_loader, val_loader, num_epochs = num_epochs, learning_rate = learning_rate, return_loss=return_loss)
    return vae, train_loss, val_loss
  else:
    vae = vae_train(vae, train_loader, val_loader, num_epochs = num_epochs, learning_rate = learning_rate, return_loss=return_loss)
    return vae



def var_encode_data(vae, data_loader):
  # Encode the synthetic time series data using the trained vae
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      output, encoded_means, encoded_vars = vae(inputs)
      encoded_means = encoded_means.detach().numpy()
      encoded_vars = encoded_vars.detach().numpy()
      encoded_data.append(np.concatenate([encoded_means, encoded_vars], axis=1))

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def var_encode(vae, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :]
  
  ds = VAE_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = var_encode_data(vae, train_loader)
  
  return(encoded_data)


def var_encode_decode_data(vae, data_loader):
  # Encode the synthetic time series data using the trained vae
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      decoded, _, _ = vae(inputs)



      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def var_encode_decode(vae, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :]
  
  ds = VAE_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = var_encode_decode_data(vae, train_loader)
  
  return(encoded_decoded_data)
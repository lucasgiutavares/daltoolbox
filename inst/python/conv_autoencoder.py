import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CAE_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]

class CAE(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(CAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64, encoding_size))
            
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 64),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 1)),
            nn.ConvTranspose1d(64, input_size, kernel_size=1),
            nn.Sigmoid()
            )
    
    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

    
# Create the cae
def cae_create(input_size, encoding_size):
  input_size = int(input_size)
  encoding_size = int(encoding_size)
  
  cae = CAE(input_size, encoding_size)
  cae = cae.float()
  return cae  

# Train the cae
def cae_train(cae, train_loader, num_epochs = 1000, learning_rate = 0.001):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(cae.parameters(), lr=learning_rate)

  for epoch in range(num_epochs):
      running_loss = 0.0
      for data in train_loader:
          inputs, _ = data
          inputs = inputs.float()
          #inputs = inputs.view(inputs.size(0), -1)
          optimizer.zero_grad()
          output = cae(inputs)
          loss = criterion(output, inputs)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
#      if (epoch + 1) % 100 == 0:
#          print('Epoch {} Loss: {:.4f}'.format(epoch+1, running_loss/len(train_loader)))

  return cae

def cae_fit(cae, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)
  
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = CAE_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  cae = cae_train(cae, train_loader, num_epochs = 1000, learning_rate = 0.001)
  
  return cae


def cae_encode_data(cae, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = cae.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def conv_encode(cae, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = CAE_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = cae_encode_data(cae, train_loader)
  
  return(encoded_data)


def cae_encode_decode_data(cae, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = cae.encoder(inputs)
      decoded = cae.decoder(encoded)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def conv_encode_decode(cae, data, batch_size = 32):
  array = data.to_numpy()
  array = array[:, :, np.newaxis]
  
  ds = CAE_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = cae_encode_decode_data(cae, train_loader)
  
  return(encoded_decoded_data)
  

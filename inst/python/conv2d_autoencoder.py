import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CAE2D_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index-1], self.data[index-1]

class CAE2D(nn.Module):
    def __init__(self, input_size, encoding_size, filter_size=1, kernel_size=4, padding=0, stride=1):
        super(CAE2D, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,1,kernel_size=5),
            nn.ReLU(True))
    
    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

    
# Create the cae
def cae2d_create(input_size, encoding_size):
  input_size = tuple(input_size)
  encoding_size = int(encoding_size)
  
  cae2d = CAE2D(input_size, encoding_size)
  cae2d = cae2d.float()
  return cae2d  

# Train the cae
def cae2d_train(cae2d, train_loader, num_epochs = 1000, learning_rate = 0.001):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(cae2d.parameters(), lr=learning_rate)
  
  for epoch in range(num_epochs):
      running_loss = 0.0
      for data in train_loader:
          inputs, _ = data
          inputs = inputs.float()
          optimizer.zero_grad()
          output = cae2d(inputs)
          loss = criterion(output, inputs)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
#      if (epoch + 1) % 100 == 0:
#          print('Epoch {} Loss: {:.4f}'.format(epoch+1, running_loss/len(train_loader)))

  return cae2d

def cae2d_fit(cae2d, data, batch_size = 20, num_epochs = 50, learning_rate = 0.001):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)
  
  ds = CAE2D_TS(data)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  cae2d = cae2d_train(cae2d, train_loader, num_epochs = num_epochs, learning_rate = learning_rate)
  
  return cae2d


def cae_encode_data(cae2d, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = cae2d.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def conv2d_encode(cae2d, data, batch_size = 32):
  array = data[:, :, np.newaxis]
  
  ds = CAE2D_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = cae2d_encode_data(cae, train_loader)
  
  return(encoded_data)


def cae2d_encode_decode_data(cae2d, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      print(inputs.shape)
      encoded = cae2d.encoder(inputs)
      decoded = cae2d.decoder(encoded)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def conv2d_encode_decode(cae2d, data, batch_size = 32):
  ds = CAE2D_TS(data)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = cae2d_encode_decode_data(cae2d, train_loader)
  
  return(encoded_decoded_data)
  

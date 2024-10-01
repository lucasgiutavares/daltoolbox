import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            #nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            #nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x
     

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x
     


class C2DEN_TS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index-1], self.data[index-1]

class C2DEN(nn.Module):
    def __init__(self, input_size, encoding_size, filter_size=1, kernel_size=4, padding=0, stride=1):
        super(C2DEN, self).__init__()
        
        self.encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
        self.decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)
        
    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

    
# Create the cae
def c2den_create(input_size, encoding_size):
  input_size = tuple(input_size)
  encoding_size = int(encoding_size)
  
  c2den = C2DEN(input_size, encoding_size)
  c2den = c2den.float()
  return c2den  

# Train the cae
def c2den_train(c2den, train_loader, num_epochs = 1000, learning_rate = 0.001):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(c2den.parameters(), lr=learning_rate)
  
  for epoch in range(num_epochs):
      running_loss = 0.0
      for data in train_loader:
          inputs, _ = data
          inputs = inputs.float()
          optimizer.zero_grad()
          output = c2den(inputs)
          loss = criterion(output, inputs)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
#      if (epoch + 1) % 100 == 0:
#          print('Epoch {} Loss: {:.4f}'.format(epoch+1, running_loss/len(train_loader)))

  return c2den

def c2den_fit(c2den, data, batch_size = 20, num_epochs = 50, learning_rate = 0.001):
  batch_size = int(batch_size)
  num_epochs = int(num_epochs)
  
  ds = C2DEN_TS(data)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  c2den = c2den_train(c2den, train_loader, num_epochs = num_epochs, learning_rate = learning_rate)
  
  return c2den


def c2den_encode_data(cae2d, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      encoded = cae2d.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())

  encoded_data = np.concatenate(encoded_data, axis=0)

  return encoded_data

def conv2den_encode(c2den, data, batch_size = 32):
  array = data[:, :, np.newaxis]
  
  ds = C2DEN_TS(array)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_data = c2den_encode_data(c2den, train_loader)
  
  return(encoded_data)


def c2den_encode_decode_data(c2den, data_loader):
  # Encode the synthetic time series data using the trained cae
  encoded_decoded_data = []
  for data in data_loader:
      inputs, _ = data
      inputs = inputs.float()
      print(inputs.shape)
      encoded = c2den.encoder(inputs)
      decoded = c2den.decoder(encoded)
      encoded_decoded_data.append(decoded.detach().numpy())

  encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)

  return encoded_decoded_data


def conv2d_encode_decode(c2den, data, batch_size = 32):
  ds = C2DEN_TS(data)
  train_loader = DataLoader(ds, batch_size=batch_size)
  
  encoded_decoded_data = c2den_encode_decode_data(c2den, train_loader)
  
  return(encoded_decoded_data)
  

# -*- coding: utf-8 -*-
"""
Denoising Autoencoder (DNS AE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DNS_AutoencoderTS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]


class DNS_AE(nn.Module):
    ##NN architecture for the autoencoder
    def __init__(self, input_size, encoding_size):
        super(DNS_AE, self).__init__()
        
        #Encode NN
        self.encoder == nn.Sequential(
            nn.Linear(input_size, 64), #NN Input layer
            nn.ReLU(True), #NN Hidden layer
            nn.Linear(64, encoding_size) ##NN Output layter
            )
        
        #Decode NN
        self.decoder == nn.Sequential(
            nn.Linear(encoding_size, 64), #NN Input layer
            nn.ReLU(True), #NN Hidden layer
            nn.Linear(64, input_size) ##NN Output layter
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


#Create denoising autoencoder (DNS AE)
def add_noise(data, noise_factor=0.3):
    data = np.array(data)
    noisy = (np.random.normal(0, 1, len(data)))
    noisy = data + noisy * noise_factor
    return noisy


def dns_ae_create(input_size, encoding_size):
    input_size = int(input_size)
    encoding_size = int(encoding_size)
    
    autoencoder = DNS_AE(input_size, encoding_size)
    autoencoder.float()
    
    return autoencoder


#Train DNS AE
def dns_ae_train(autoencoder, train_loader, num_epochs = 1000, learning_rate = 0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.float()
            inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    return autoencoder


def dns_ae_fit(autoencoder, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001, noise_factor=0.3):
    batch_size = int(batch_size)
    num_epochs = int(num_epochs)
    
    #Add noise to data before train
    array = add_noise(data, noise_factor)
    #array = data.to_numpy() #The add_noise function already transforms data into a numpy array
    array = array[:, :, np.newaxis]
    
    ds = DNS_AutoencoderTS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)
    
    autoencoder = dns_ae_train(autoencoder, train_loader, num_epochs = num_epochs, learning_rate = learning_rate)
    
    autoencoder = None
    return autoencoder


def encode_data(autoencoder, data_loader):
    # Encode the synthetic time series data using the trained autoencoder
    encoded_data = []
    for data in data_loader:
        inputs, _ = data
        inputs = inputs.float()
        inputs = inputs.view(inputs.size(0), -1)
        encoded = autoencoder.encoder(inputs)
        encoded_data.append(encoded.detach().numpy())
        
    encoded_data = np.concatenate(encoded_data, axis=0)
    
    return encoded_data


def autoencoder_encode(autoencoder, data, batch_size = 32):
    array = data.to_numpy()
    array = array[:, :, np.newaxis]
    
    ds = DNS_AutoencoderTS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)
    
    encoded_data = encode_data(autoencoder, train_loader)
    
    return(encoded_data)


def encode_decode_data(autoencoder, data_loader):
    # Encode the synthetic time series data using the trained autoencoder
    encoded_decoded_data = []
    for data in data_loader:
        inputs, _ = data
        inputs = inputs.float()
        inputs = inputs.view(inputs.size(0), -1)
        encoded = autoencoder.encoder(inputs)
        decoded = autoencoder.decoder(encoded)
        encoded_decoded_data.append(decoded.detach().numpy())
        
    encoded_decoded_data = np.concatenate(encoded_decoded_data, axis=0)
    
    return encoded_decoded_data


def autoencoder_encode_decode(autoencoder, data, batch_size = 32):
    array = data.to_numpy()
    array = array[:, :, np.newaxis]
    
    ds = DNS_AutoencoderTS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)
    
    encoded_decoded_data = encode_decode_data(autoencoder, train_loader)
    
    return(encoded_decoded_data)

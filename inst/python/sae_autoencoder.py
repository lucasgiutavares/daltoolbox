# -*- coding: utf-8 -*-
"""
Stacked Autoencoder (SAE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SAE_AutoencoderTS(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = np.random.randn(num_samples, input_size)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index]


class SAE(nn.Module):
    ##NN architecture for the autoencoder
    def __init__(self, input_size, encoding_size):
        super(SAE, self).__init__()
        
        #Encode NN
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64), #NN Input layer
            nn.ReLU(True), #NN Hidden layer
            nn.Linear(64, encoding_size) ##NN Output layter
            )
        
        #Decode NN
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 64), #NN Input layer
            nn.ReLU(True), #NN Hidden layer
            nn.Linear(64, input_size) ##NN Output layter
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


#Create Stack of Autoencoders
def sae_create(input_size, encoding_size, k_ae=3):
    input_size = int(input_size)
    encoding_size = int(encoding_size)
    k_ae = int(k_ae)
    
    #Stack of k autoencoders
    stack = []
    for k in range(k_ae):
        stack.append(SAE(input_size, encoding_size))
        stack[k].float()
        print(f'Autoencoder layer {k} added to the stack')
    
    return stack


#Train SAE
def sae_train(autoencoder, train_loader, num_epochs = 1000, learning_rate = 0.001):
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


def sae_fit(stack, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001):
    batch_size = int(batch_size)
    num_epochs = int(num_epochs)

    #STEP 1 - Start fitting first k autoencoder using original input layer    
    array = data.to_numpy()
    array = array[:, :, np.newaxis]
    
    ds = SAE_AutoencoderTS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)

    print(f'Fit ae_k{0}')
    ae_k1 = stack[0]
    ae_k1 = sae_train(ae_k1, train_loader, num_epochs = num_epochs, learning_rate = learning_rate)
    ae_k_out = sae_encode_decode(ae_k1, data)
    
    #STEP 2 - Fit internal layers using outputs from previous layers
    internal = int(len(stack)-1)
    
    for k in range(1, internal):
        print(f'Fit ae_k{k}')
        
        #ds = SAE_AutoencoderTS(ae_k_out)
        ds = ae_k_out
        train_loader = DataLoader(ds, batch_size=batch_size)
        
        ae_k = stack[k]
        ae_k = sae_train(ae_k, train_loader, num_epochs = num_epochs, learning_rate = learning_rate)
        ae_k_out = sae_encode_decode(ae_k, ae_k_out)
    
    #STEP 3 - Fit last k autoencoder
    print(f'Fit ae_k{len(stack)}')
    ds = SAE_AutoencoderTS(ae_k_out)
    train_loader = DataLoader(ds, batch_size=batch_size)
    autoencoder = stack[-1]
    autoencoder = sae_train(autoencoder, train_loader, num_epochs = num_epochs, learning_rate = learning_rate)
    
    return autoencoder


def sae_encode_data(autoencoder, data_loader):
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


def sae_encode(autoencoder, data, batch_size = 32):
    array = data.to_numpy()
    array = array[:, :, np.newaxis]
    
    ds = SAE_AutoencoderTS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)
    
    encoded_data = sae_encode_data(autoencoder, train_loader)
    
    return(encoded_data)


def sae_encode_decode_data(autoencoder, data_loader):
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


def sae_encode_decode(autoencoder, data, batch_size = 32):
    array = data.to_numpy()
    array = array[:, :, np.newaxis]
    
    ds = SAE_AutoencoderTS(array)
    train_loader = DataLoader(ds, batch_size=batch_size)
    
    encoded_decoded_data = sae_encode_decode_data(autoencoder, train_loader)
    
    return(encoded_decoded_data)

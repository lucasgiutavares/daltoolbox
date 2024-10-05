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


class DNS_Autoencoder(Dataset):
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
    
    autoencoder = None
    return autoencoder

#Train DNS AE
def dns_ae_train(autoencoder, train_loader, num_epochs = 1000, learning_rate = 0.001):
    autoencoder = None
    return autoencoder


def dns_ae_fit(autoencoder, data, batch_size = 32, num_epochs = 1000, learning_rate = 0.001):
    autoencoder = None
    return autoencoder


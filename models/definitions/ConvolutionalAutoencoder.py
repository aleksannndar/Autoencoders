from matplotlib.cbook import flatten
import torch.nn.functional as F
import torch.nn as nn
import torch

# (W-F + 2P)/S + 1

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, imgShape=(1,28,28)):
        super().__init__()
        
        flattenSize = (int) ((((((imgShape[1] - 2)/2 + 1) - 2)/2 + 1) - 3)/2 + 1)

        self.encoder = nn.Sequential(
            nn.Conv2d(imgShape[0], 32, 4, stride=2, padding=1), # -> 32, 14, 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4,stride = 2, padding = 1), # -> 64, 7, 7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride = 2, padding = 0), # -> 128, 3, 3
            nn.Flatten(),
            nn.Linear(128*flattenSize*flattenSize, 70),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(70, 128*flattenSize*flattenSize),
            nn.ReLU(),
            nn.Unflatten(1,(128, 3, 3)),
            nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 0, output_padding=1), #for CIFAR10 add output_padding = 1
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, imgShape[0], 4, stride = 2, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
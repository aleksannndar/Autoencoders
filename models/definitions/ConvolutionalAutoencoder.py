import torch.nn.functional as F
import torch.nn as nn
import torch

# (W-F + 2P)/S + 1

class ConvolutionalAutoencoderMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), # -> 32, 14, 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4,stride = 2, padding = 1), # -> 64, 7, 7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride = 2, padding = 0), # -> 128, 3, 3
            nn.Flatten(),
            nn.Linear(1152, 10),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 1152),
            nn.ReLU(),
            nn.Unflatten(1,(128, 3, 3)),
            nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride = 2, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
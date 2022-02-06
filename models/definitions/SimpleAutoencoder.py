import torch.nn as nn
import torch.nn.functional as F

class SimpleAutoencoderMNIST(nn.Module):
    def __init__(self, imgShape=(28,28)):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(imgShape[0]*imgShape[1], 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,18),
            nn.ReLU(),
            nn.Linear(18,9)
        )

        self.decoder = nn.Sequential(
            nn.Linear(9,18),
            nn.ReLU(),
            nn.Linear(18,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,imgShape[0]*imgShape[1]),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from models.definitions import ConvolutionalAutoencoder
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plotInputOutput(outputs, numEpochs):
    for i in range(0, numEpochs):
        plt.figure(figsize=(9,2))
        plt.gray()
        imgs = outputs[i][1].detach().cpu().numpy()
        recon = outputs[i][2].detach().cpu().numpy()
        for j, item in enumerate(imgs):
            if j >= 9: break
            plt.subplot(2, 9, j+1)
            plt.imshow(item.reshape(28,28))
            
        for j, item in enumerate(recon):
            if j >= 9: break
            plt.subplot(2, 9, 9+j+1)
            plt.imshow(item.reshape(28,28))
    plt.show()

def trainConvolutionalAutoencoder(numEpochs):
    if(torch.cuda.is_available()):
        print("Running on GPU")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.ToTensor()
    mnistData = datasets.MNIST(root="./data/trainingData", train = True, download=True, transform=transform)
    dataLoader = DataLoader(dataset= mnistData,
                            batch_size=64,
                            shuffle=True)
    
    cAutoencoder = ConvolutionalAutoencoder.ConvolutionalAutoencoderMNIST().to(device)
    cOptim = torch.optim.Adam(cAutoencoder.parameters(),
                                lr=1e-3,
                                weight_decay=1e-5)
    lossFunction = nn.MSELoss()

    outputs = []
    for epoch in tqdm(range(numEpochs)):
        for (img, _) in dataLoader:
            img = img.to(device)
            recon = cAutoencoder(img)
            loss = lossFunction(recon, img)

            cOptim.zero_grad()
            loss.backward()
            cOptim.step()
        print(f"Epoch:{epoch+1}, Loss:{loss.item():.4f}")
        outputs.append((epoch, img, recon))
    plotInputOutput(outputs, numEpochs)
    
trainConvolutionalAutoencoder(10)
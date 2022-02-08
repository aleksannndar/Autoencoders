import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

transform = transforms.ToTensor()

cifarData = datasets.CIFAR10(root="./data/trainingData", train = True, download=True, transform=transform)
dataLoader = DataLoader(dataset=cifarData,
                        batch_size=64,
                        shuffle=True)

dataIter = iter(dataLoader)
images, labels = dataIter.next()
for (images, _) in dataIter:
    print(images[0].reshape(3*32*32).reshape(-1,32,32).size())
    plt.imshow(images[0].permute(1,2,0))
    plt.show()
    break
print(torch.min(images), torch.max(images))
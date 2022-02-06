import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

cifarData = datasets.CIFAR10(root="./data/trainingData", train = True, download=True, transform=transform)
dataLoader = DataLoader(dataset=cifarData,
                        batch_size=64,
                        shuffle=True)

dataIter = iter(dataLoader)
images, labels = dataIter.next()
for (images, _) in dataIter:
    print(images.size())
    break
print(torch.min(images), torch.max(images))
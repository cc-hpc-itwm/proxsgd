import sys
import torch
import torch.nn as nn
import config
from torchvision import datasets, transforms


#Load CIFAR100 dataset
def load_cifar100():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = torch.utils.data.DataLoader(datasets.CIFAR100(root='dataset', train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize,]), download=True),batch_size=config.batch_size,num_workers=6, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR100(root='dataset', train=False, transform=transforms.Compose([transforms.ToTensor(),normalize,])),batch_size=config.batch_size, num_workers=4, shuffle=False)
    
    return train_data, test_loader
#Load MNIST dataset
def load_mnist():
    train_data= torch.utils.data.DataLoader(datasets.MNIST(root='dataset',train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), download=True),
                                            batch_size=config.batch_size, num_workers=6, shuffle=True)
    test_loader= torch.utils.data.DataLoader(datasets.MNIST(root='dataset',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), download=True),
                                            batch_size=config.batch_size, num_workers=4, shuffle=False)
    return train_data, test_loader

if __name__ == "__main__":
    train_data, test_loader=load_mnist()
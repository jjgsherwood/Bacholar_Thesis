from config import *

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class UniformNoise(object):

    def __call__(self, x):
        with torch.no_grad():
            noise = x.new().resize_as_(x).uniform_()
            x = x * 255 + noise
            x = x / 256
        return x

    def __repr__(self):
        return "UniformNoise"

def load_mnist(data_dir, batch_size):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        UniformNoise(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        UniformNoise(),
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, transform=train_transform, download=True)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=test_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size, drop_last=True,
                              shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size, drop_last=True,
                             shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    return train_loader, test_loader

def load_liver():
    pass

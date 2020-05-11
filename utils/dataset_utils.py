from utils.config import *

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import copy
import random
import os
import configparser

import numpy as np
from sklearn.model_selection import train_test_split

class UniformNoise():
    def __call__(self, x):
        with torch.no_grad():
            noise = x.new().resize_as_(x).uniform_()
            x = x * 255 + noise
            x = x / 256
        return x

    def __repr__(self):
        return "UniformNoise"

class From_2D_to_3D():
    def __call__(self, x):
        return x.unsqueeze_(-1)

    def __repr__(self):
        return "Unsqueeze"

class TensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data, transform=None):
        assert data.size(0) == data.size(0)
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x,

    def __len__(self):
        return self.data.size(0)

def load_mnist(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        UniformNoise(),
        From_2D_to_3D()
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size, drop_last=True,
                              shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size, drop_last=True,
                             shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    return train_loader, test_loader

def split_data(dataset, test_size=TEST_RATIO, seed=42):
    np.random.seed(seed)
    train, test = train_test_split(dataset.data, test_size=test_size)
    train_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)

    train_dataset.data = train
    test_dataset.data = test
    return train_dataset, test_dataset

def create_dataset_1x1(data):
    return data.reshape(-1, 1, 1, 1, data.shape[-1])

def load_liver(data_file, batch_size):
    assert os.path.isfile(data_file)

    transform = transforms.Compose([
        UniformNoise()
    ])

    data = np.load(data_file, 'r')
    data = create_dataset_1x1(data)

    tensor_data = torch.Tensor(data)
    dataset = TensorDataset(tensor_data, transform)

    train_dataset, test_dataset = split_data(dataset)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)

    return train_loader, test_loader

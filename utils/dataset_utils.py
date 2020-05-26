from utils.config import *

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, SequentialSampler, BatchSampler

import copy
import random
import os
import configparser

import numpy as np
from sklearn.model_selection import train_test_split

class UniformNoise():
    def __init__(self, max_bit=8):
        self.max = 2**max_bit

    def __call__(self, x):
        with torch.no_grad():
            noise = x.new().resize_as_(x).uniform_()
            x = x * (self.max - 1) + noise
            x = x / self.max
        return x

    def __repr__(self):
        return "UniformNoise"

class Vector_unit_normalization():
    def __call__(self, x):
        return x / np.sqrt((x**2).sum(axis=3))

    def __repr__(self):
        return "Vector unit normalization"

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

def load_liver(data, batch_size):
    transform = transforms.Compose([
        Vector_unit_normalization(),
        UniformNoise(12)
    ])

    data = create_dataset_1x1(data)

    tensor_data = torch.Tensor(data)
    dataset = TensorDataset(tensor_data, transform)

    train_dataset, test_dataset = split_data(dataset)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)

    return train_loader, test_loader

def load_liver_all(data):
    transform = transforms.Compose([
        Vector_unit_normalization(),
    ])

    data = create_dataset_1x1(data)
    return torch.Tensor(data)

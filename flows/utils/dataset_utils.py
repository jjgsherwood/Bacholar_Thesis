import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import copy
import random

import numpy as np
from sklearn.model_selection import train_test_split


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
                              shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size, drop_last=True,
                             shuffle=False, pin_memory=True, num_workers=2)

    return train_loader, test_loader


def split_ImageFolder(dataset, test_size=0.2, seed=42):
    np.random.seed(seed)
    samples = dataset.samples
    train, test = train_test_split(samples, test_size=test_size)
    train_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)

    train_dataset.samples = train
    train_dataset.imgs = train

    test_dataset.samples = test
    test_dataset.imgs = test
    return train_dataset, test_dataset


def load_celeba(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        UniformNoise(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_dataset, test_dataset = split_ImageFolder(dataset)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             drop_last=True, pin_memory=True, num_workers=2)
    return train_loader, test_loader

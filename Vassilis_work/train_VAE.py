import os
import time
from datetime import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from model import *

def train(config):
    # Initialize the device which to run the model on
    use_cuda = torch.cuda.is_available()
    device = torch.device(config.device if use_cuda else 'cpu')

    # Initialize the dataset and data loader (note the +1)
    dataset = RamanDataset(config.seq_length, config.mode, config.folder)
    data_loader = DataLoader(dataset, config.batch_size, shuffle=True)

    # Initialize the model that we are going to use
    parameters = {
        'device': device,
        'input_dim': dataset.sequence_len,
        'hidden_dims': [512, 256],
        'z_dim': 32,
    }

    model = VAEModel('MLP', **parameters).to(device)

    # Setup the loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = LogCoshLoss()
    # loss_function = nn.MSELoss()
    print("start training")

    for epoch in range(config.epochs):
        for batch_inputs, batch_targets in tqdm(data_loader, desc=f"Training epoch {epoch}", leave=False, position=0):
            # Move to GPU
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            # Reset for next iteration
            model.zero_grad()

            # Forward pass
            mean, log_std, out = model(batch_inputs)

            # Compute the loss, gradients and update network parameters
            loss = loss_function(out.float(), batch_inputs.float()) + KLD(mean, log_std).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config.max_norm)

            optimizer.step()
            print(loss)

        out = out.cpu().detach().numpy()
        batch_inputs = batch_inputs.cpu().detach().numpy()
        batch_targets = batch_targets.cpu().detach().numpy()
        plt.figure()
        plt.plot(range(out.shape[1]), out[0], label="LSTM")
        plt.plot(range(batch_inputs.shape[1]), batch_inputs[0], label="raw")
        plt.plot(range(batch_targets.shape[1]), batch_targets[0], label='manualy')
        plt.legend()
        plt.savefig(f"epoch {epoch}")
        plt.close()

    print('Done training.')

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--folder', type=str, default=None,
                        help="Path to the folder off the dataset")
    parser.add_argument('--mode', type=str, default="smooth",
                        help="Structure of the NN output [all, smooth, split].")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=256,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=3,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')

    parser.add_argument('--epochs', type=int, default=int(100),
                        help='Number of training epochs')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--device', type=str, default="cuda",
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)

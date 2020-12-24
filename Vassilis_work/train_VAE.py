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

from utils import RamanDataset
from model import *

def train(config):
    # Initialize the device which to run the model on
    use_cuda = torch.cuda.is_available()
    device = torch.device(config.device if use_cuda else 'cpu')

    # Initialize the dataset and data loader (note the +1)
    dataset = RamanDataset(config.seq_length, config.mode, config.folder)
    data_loader = DataLoader(dataset, config.batch_size, shuffle=True)
    print(dataset.shape)
    # Initialize the model that we are going to use
    parameters = {
        'device': device,
        'input_dim': dataset.shape
    }

    model = VAEModel().to(device)

    # Setup the loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = LogCoshLoss()
    # loss_function = nn.MSELoss()
    print("start training")

    for i in range(config.epochs):
        # Only for time measurement of step through network
        t1 = time.time()
        for batch_inputs, batch_targets in tqdm(data_loader):
            # Move to GPU
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            # Reset for next iteration
            model.zero_grad()

            # Forward pass
            batch_inputs = batch_inputs.unsqueeze(2)
            out, _ = model(batch_inputs)
            out = out.squeeze()

            # Compute the loss, gradients and update network parameters
            loss = loss_function(out.float(), batch_targets.float())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config.max_norm)

            optimizer.step()
            # print(loss)

        # Just for time measurement
        t2 = time.time()

        eps = len(dataset)/float(t2-t1)
        t = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"{t} Epoch {i}/{config.epochs}, Examples/Sec = {eps:.2f}, \
                loss = {loss*10000:.2f}")

        out = out.cpu().detach().numpy()
        batch_inputs = batch_inputs.cpu().detach().numpy()
        batch_targets = batch_targets.cpu().detach().numpy()
        plt.figure()
        plt.plot(range(out.shape[1]), out[0], label="LSTM")
        plt.plot(range(batch_inputs.shape[1]), batch_inputs[0], label="raw")
        plt.plot(range(batch_targets.shape[1]), batch_targets[0], label='manualy')
        plt.legend()
        plt.savefig(f"epoch {i}")
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

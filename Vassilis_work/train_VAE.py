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
from torchvision.utils import make_grid, save_image

from utils import *
from model import *

def sample_and_save(model, epoch, batch_size=64):
    """
    Function that generates and saves samples from the VAE.  The generated
    samples and mean images should be saved, and can eventually be added to a
    TensorBoard logger if wanted.
    Inputs:
        model - The VAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
    """
    sample = model.sample(batch_size)
    grid = make_grid(sample.cpu(), nrow=8)
    save_image(grid, os.path.join('images', f'sample_{epoch}.png'))

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
        'hidden_dims': [1024, 512, 256, 128],
        'z_dim': 32,
    }

    model = VAEModel('MLP', **parameters).to(device)

    # Setup the loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # loss_function = LogCoshLoss()
    loss_function = nn.MSELoss('none')

    print("start training")
    loss_graph = []
    loss_graph_rec = []
    loss_graph_reg = []

    for epoch in range(config.epochs):
        loss_graph_tmp = 0
        loss_graph_reg_tmp = 0
        loss_graph_rec_tmp = 0

        for batch_inputs, batch_targets in tqdm(data_loader, desc=f"Training epoch {epoch}", leave=False, position=0):
            # Move to GPU
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            # Reset for next iteration
            model.zero_grad()

            # Forward pass
            mean, log_std, out = model(batch_inputs)

            # Compute the loss, gradients and update network parameters
            rec_loss, reg_loss = loss_function(out.float(), batch_inputs.float()).sum(1).mean(0), KLD(mean, log_std).mean()
            loss = rec_loss + reg_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config.max_norm)

            optimizer.step()

            loss_graph_tmp += loss
            loss_graph_reg_tmp += reg_loss
            loss_graph_rec_tmp += rec_loss

        loss_graph.append(loss_graph_tmp / epoch)
        loss_graph_rec.append(loss_graph_rec_tmp / epoch)
        loss_graph_reg.append(loss_graph_reg_tmp / epoch)

        if not epoch % 5:
            plt.figure()
            plt.plot(range(len(loss_graph)), loss_graph, label='total')
            plt.plot(range(len(loss_graph)), loss_graph_rec, label='reconstruction')
            plt.plot(range(len(loss_graph)), loss_graph_reg, label='regularization')
            plt.legend()
            plt.savefig(f"images/loss", dpi=500)
            plt.close()

            out = out.cpu().detach().numpy()
            batch_inputs = batch_inputs.cpu().detach().numpy()
            batch_targets = batch_targets.cpu().detach().numpy()
            plt.figure()
            plt.plot(range(out.shape[1]), out[0], label="LSTM")
            plt.plot(range(batch_inputs.shape[1]), batch_inputs[0], label="raw")
            plt.plot(range(batch_targets.shape[1]), batch_targets[0], label='manualy')
            plt.legend()
            plt.savefig(f"images/epoch {epoch}", dpi=500)
            plt.close()

            sample_and_save(model, epoch, 64)

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

    parser.add_argument('--epochs', type=int, default=int(250),
                        help='Number of training epochs')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--device', type=str, default="cuda",
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)

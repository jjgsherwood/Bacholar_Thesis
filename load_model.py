from argparse import ArgumentParser
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import configparser
import pickle
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from utils.dataset_utils import load_liver, load_mnist
from utils.train_utils import train_nll, test_nll, Warmup
from utils.image_utils import viz_array_grid, viz_array_set_grid, save_image

from glow3D import GLOW

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./experiment_output')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'model_parameters.pkl'), 'rb') as f:
        args = pickle.load(f)

    print("Args:")
    for k, v in vars(args).items():
        print("  {}={}".format(k, v))
    print(flush=True)

    if args.dataset == 'mnist':
        train_loader, test_loader = load_mnist(args.data_dir, args.batch_size)
        num_channels = 1
    elif args.dataset == 'liver':
        train_loader, test_loader = load_liver(args.data_dir, args.batch_size)
        num_channels = 1

    _use_cuda = torch.cuda.is_available() and args.cuda
    if _use_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if _use_cuda else 'cpu')

    model = GLOW(args.k, args.l, num_channels, args.hidden_ch, args.structure).to(device)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.pt')))
    model.eval()

    if args.dataset == 'liver':
        x = next(iter(test_loader))[0]
        x = x.to(device)

        with torch.no_grad():
            z, _ = model(x)
            x_rec, _ = model.inverse(z)
            z_samples = torch.randn(*z.size()).to(device)
            x_gen, _ = model.inverse(z_samples)

        x = x.detach().cpu().squeeze().numpy()
        x_rec = x_rec.detach().cpu().squeeze().numpy()
        x_gen = x_gen.detach().cpu().squeeze().numpy()

        for x_o, x_r, x_g in zip(x, x_rec, x_gen):
            plt.plot(x_o, label='orginal')
            plt.plot(x_r, label='recreated')
            plt.plot(x_g, label='sampled')
            plt.legend()
            plt.show()
            print(x_o.shape, x_r.shape, x_g.shape)
            break

    elif args.dataset == 'mnist':
        x = next(iter(test_loader))[0][:16]
        x = x.to(device)

        with torch.no_grad():
            z, _ = model(x)
            x_rec, _ = model.inverse(z)
            z_samples = torch.randn(*z.size()).to(device)
            x_gen, _ = model.inverse(z_samples)

        x = x.detach().cpu().squeeze(-1).numpy()
        x_rec = x_rec.detach().cpu().squeeze(-1).numpy()
        x_gen = x_gen.detach().cpu().squeeze(-1).numpy()

        # samples
        img = viz_array_grid(x_gen, 4, 4, padding=2)
        save_image(img, os.path.join(args.output_dir, 'samples.png'), (8, 8))

        # reconstruction
        img = viz_array_set_grid([[x, x_rec]], 4, 4, padding=2)
        save_image(img, os.path.join(args.output_dir, 'reconstruction.png'), (16, 8))
        print('Images are saved')

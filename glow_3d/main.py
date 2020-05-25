from argparse import ArgumentParser
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import configparser
import pickle
import matplotlib.pyplot as plt
import random

import torch
import torch.optim as optim

from utils.dataset_utils import load_liver, load_mnist
from utils.train_utils import train_nll, test_nll, Warmup
from utils.image_utils import viz_array_grid, viz_array_set_grid, save_image

from glow3D import GLOW


if __name__ == "__main__":
    # read config to find structure choices
    config = configparser.ConfigParser()
    config.read('glow_config.ini')
    structure_choices = config.sections()

    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--cuda', type=eval, default=True, required=False, choices=[True, False])
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'liver'])
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--log_step', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./experiment_output')

    # optimizer
    parser.add_argument('--warmup_steps', type=int, default=3000)
    parser.add_argument('--init_lr', type=float, default=1e-3)

    # glow
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--hidden_ch', type=int, default=1)
    parser.add_argument('--structure', type=str, default=structure_choices[0],  choices=structure_choices)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
    with open(os.path.join(args.output_dir, 'model_parameters.pkl'), 'wb') as f:
        pickle.dump(args, f)

    parameters = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, betas=(0.9, 0.9999))
    scheduler = Warmup(optimizer, args.init_lr, args.warmup_steps)

    ############################################
    # Training
    ############################################
    print('Training', flush=True)

    best_loss = np.inf

    for epoch in range(args.epochs):
        # train
        print('-'*50)
        print('Epoch {:3d}/{:3d}'.format(epoch+1, args.epochs))
        start_time = datetime.now()
        train_nll(model, optimizer, train_loader, scheduler, args.log_step, device)
        end_time = datetime.now()
        time_diff = relativedelta(end_time, start_time)
        print('Elapsed time: {}h {}m {}s'.format(time_diff.hours, time_diff.minutes, time_diff.seconds))
        loss = test_nll(model, test_loader, device)
        print('Validation| bits: {:2.2f}'.format(loss), flush=True)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
            print('Model is saved')

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

                plt.plot(x[0], label='orginal')
                plt.plot(x_rec[0], label='recreated')
                plt.plot(x_gen[0], label='sampled')
                plt.legend()
                strFile = os.path.join(args.output_dir, f'sample.png')
                if os.path.isfile(strFile):
                    os.remove(strFile)
                plt.savefig(strFile)
                plt.clf()

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

    print('Training is finished')

from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np


class Warmup(object):
    def __init__(self, optimizer, init_lr, warmup_steps):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_steps = float(warmup_steps)
        self.last_batch = 0
        self.lr = init_lr

    def step(self):
        self.last_batch += 1
        if self.last_batch < self.warmup_steps:
            mult = self.last_batch / max(self.warmup_steps, 1)
            self.lr = self.init_lr * mult
        else:
            mult = (self.warmup_steps / self.last_batch)**0.5
            self.lr = self.init_lr * mult
            self.lr = max(self.lr, self.init_lr / 10)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

def get_bits_per_dim(x, model):
    z, log_p = model(x)
    dist = Normal(torch.zeros_like(z, device=z.device), 1.0)
    # print(log_p.sum().cpu().detach().numpy(), dist.log_prob(z).sum().to(z.device).cpu().detach().numpy(), ((x - model.inverse(z)[0])**2).sum().cpu().detach().numpy())
    ll = log_p.sum() + dist.log_prob(z).sum().to(z.device) - ((x - model.inverse(z)[0])**2).sum() * 10000
    ll = 8.0 - ll / (np.log(2.0) * x.nelement())
    # ll = ((x - model.inverse(z)[0])**2).sum()
    return ll

def train_nll(model, optimizer, loader, scheduler=None, log_step=None, device=torch.device('cuda')):
    model.train()

    for batch_idx, (x, *_) in enumerate(loader):
        scheduler.step()

        x = x.to(device)
        loss = get_bits_per_dim(x, model) # TODO change this for liver

        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()

        # gradient clipping
        parameters = list(filter(lambda x: x.requires_grad, model.parameters()))
        nn.utils.clip_grad_value_(parameters, 5)
        grad_norm = nn.utils.clip_grad_norm_(parameters, 100)

        optimizer.step()

        if log_step:
            if batch_idx % log_step == 0:
                print('  {}| {:5d}/{:5d}| bits: {:2.2f}, lr: {:0.5f} grad_norm: {:2.1f}'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx,
                    len(loader), loss.item(), scheduler.lr, grad_norm
                ), flush=True)

def test_nll(model, loader, device=torch.device('cuda')):
    model.eval()
    bits = 0

    with torch.no_grad():
        for x, *_ in loader:
            x = x.to(device)
            bits += get_bits_per_dim(x, model).item()

    return bits / len(loader)

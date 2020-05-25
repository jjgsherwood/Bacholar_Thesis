from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np


def train_nll(model, optimizer, loader, loss_func, log_step=None, device=torch.device('cuda')):
    model.train()

    for batch_idx, (x, *_) in enumerate(loader):
        x = x.to(device)
        loss = loss_func(x, model)

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
                print('  {}| {:5d}/{:5d}| bits: {:2.2f}, grad_norm: {:2.1f}'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx,
                    len(loader), loss.item(), grad_norm
                ), flush=True)

def test_nll(model, loader, loss_func, device=torch.device('cuda')):
    model.eval()
    bits = 0

    with torch.no_grad():
        for x, *_ in loader:
            x = x.to(device)
            bits += loss_func(x, model).item()

    return bits / len(loader)

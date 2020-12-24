import torch.nn as nn
import torch
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, lstm_num_hidden=256, lstm_num_layers=2):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.linear = nn.Linear(lstm_num_hidden * 2, 1)

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        return self.linear(h), state

class RNN2Model(nn.Module):
    def __init__(self, lstm_num_hidden=256, lstm_num_layers=2):
        super(RNN2Model, self).__init__()
        self.lstm1 = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.lstm2 = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.linear1 = nn.Linear(lstm_num_hidden * 2, 1)
        self.linear2 = nn.Linear(lstm_num_hidden * 2, 1)


    def forward(self, x, state=(None, None)):
        state1, state2 = state
        h1, state1 = self.lstm1(x, state1)
        h2, state2 = self.lstm2(x, state2)
        return torch.cat((self.linear1(h1), self.linear2(h2)), 1), (state1, state2)

class RNN3Model(nn.Module):
    def __init__(self, lstm_num_hidden=256, lstm_num_layers=2):
        super(RNN3Model, self).__init__()
        self.lstm1 = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.lstm2 = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.linear1 = nn.Linear(lstm_num_hidden * 2, 1)
        self.linear2 = nn.Linear(lstm_num_hidden * 2, 1)

    def forward(self, x, state=(None, None)):
        state1, state2 = state
        h1, state1 = self.lstm1(x, state1)
        h2, state2 = self.lstm2(x, state2)
        o1, o2 = self.linear1(h1), self.linear2(h2)
        return torch.cat((o1, o2, o1 + o2), 1), (state1, state2)

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

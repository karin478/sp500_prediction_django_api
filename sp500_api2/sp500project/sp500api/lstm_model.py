import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Your data loading and preprocessing code here

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).float()
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).float()

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        return out

input_size = 50
hidden_size = 1024
num_layers = 4
output_size = 1
dropout_rate = 0.3
bidirectional = True

model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)


def save_lstm_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_lstm_model(filepath, input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional)
    model.load_state_dict(torch.load(filepath))
    return model

def load_lstm_model4():
    input_size = 50
    hidden_size = 1024
    num_layers = 4
    output_size = 1
    dropout_rate = 0.3
    bidirectional = True

    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
    model = load_lstm_model(model_path, input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional)
    return model




"""
IDEAS:
1. start with simplest: use padded sequence from encoder to decoder
2. group sequence of same lengths (so that the model no needa encode and decode padder)
"""
import numpy as np
import torch
import torch.nn as nn

from src.common.logger import get_logger

logger = get_logger(__name__)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p, is_orthogonal_init = True):
        super(LSTMAutoEncoder, self).__init__()
        
        self.encoder = LSTMEncoder(
            input_size, hidden_size, 
            num_layers, is_orthogonal_init
        )
        self.decoder = LSTMDecoder(
            hidden_size, input_size, 
            num_layers, dropout_p,
            is_orthogonal_init
        )
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        seq_size = x.size(1)

        encoder_out = self.encoder(x)
        # (batch_sz, seq len, hidden sz)
        decoder_in = encoder_out.expand(-1, seq_size, -1)
        decder_out = self.decoder(decoder_in)

        return decder_out


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, is_orthogonal_init):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers = num_layers,
            batch_first = True
        )

        if is_orthogonal_init:
            nn.init.orthogonal_(self.lstm.weight_ih_l0, gain = np.sqrt(2))
            nn.init.orthogonal_(self.lstm.weight_hh_l0, gain = np.sqrt(2))
        
    def forward(self, x):
        device = x.device
        batch_sz = x.size(0)

        h0 = torch.zeros(
            self.num_layers, batch_sz, self.hidden_size
        )
        h0 = h0.to(device)
        c0 = h0.clone()
        
        # (bs, seq len, 4)
        out, _ = self.lstm(x, (h0, c0)) 
        hlast = out[:, -1, :]
        return hlast.unsqueeze(1)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p, is_orthogonal_init):
        super(LSTMDecoder, self).__init__()
        
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers = num_layers,
            batch_first = True
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p = dropout_p)
        
        if is_orthogonal_init:
            nn.init.orthogonal_(self.lstm.weight_ih_l0, gain = np.sqrt(2))
            nn.init.orthogonal_(self.lstm.weight_hh_l0, gain = np.sqrt(2))
        
    def forward(self, x):
        # x: (batch size, padded seq len, hidden size)
        device = x.device
        batch_sz, _, hidden_sz = x.size()
        
        h0 = torch.zeros(self.num_layers, batch_sz, hidden_sz).to(device)
        c0 = h0.clone()
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        fc_out = self.fc(self.dropout(lstm_out))
        return self.relu(fc_out)
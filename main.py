import os

from src.config import Config as config
from src.learner import Learner
from src.models.autoencoder import LSTMAutoEncoder
from src.data.dataloaders import get_dataloaders

lstm_autoencoder = LSTMAutoEncoder(
    input_size = config.INPUT_SIZE,
    hidden_size = config.HIDDEN_SIZE,
    num_layers = config.NUM_LAYERS,
    dropout_p = config.DROPOUT_P   
)
train_loader, val_loader, test_loader = get_dataloaders()

learner = Learner(lstm_autoencoder, train_loader, val_loader, config)
learner.train()
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/media/alex/Amazing/personal/Project/RICO_dataset/test_data'

RESIZE_FACTOR = 0.1
WIDTH = 1440
HEIGHT = 2560

INPUT_SIZE = 4
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.4

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01
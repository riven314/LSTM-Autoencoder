import torch

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # under which there are "train", "val", "test" subfolders
    DATA_DIR = '/media/alex/Amazing/personal/Project/RICO_dataset/test_data'

    OUTPUT_DIR = ''
    EXPERIMENT_NAME = ''

    RESIZE_FACTOR = 0.1
    WIDTH = 1440
    HEIGHT = 2560

    INPUT_SIZE = 4
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.4
    IS_LSTM_ORTHOGONAL_INIT = True

    BATCH_SIZE = 4
    EPOCH_N = 10
    SAVE_CYCLE = 3
    VERBOSE_CYCLE = 1
    ENCODER_LR = 0.1
    ENCODER_GAMMA = 0.1
    ENCODER_STEP_CYCLE = 5
    DECODER_LR = ENCODER_LR
    DECODER_GAMMA = ENCODER_GAMMA
    DECODER_STEP_CYCLE = ENCODER_STEP_CYCLE
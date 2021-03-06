import torch

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # under which there are "train", "val", "test" subfolders
    #DATA_DIR = '/media/alex/Amazing/personal/Project/RICO_dataset/test_data'
    DATA_DIR = '/userhome/34/h3558533/RICO/all_data'

    OUTPUT_DIR = 'outputs'
    EXPERIMENT_NAME = '00_prototype'

    RESIZE_FACTOR = 0.1
    WIDTH = 1440
    HEIGHT = 2560

    INPUT_SIZE = 4
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT_P = 0.
    IS_LSTM_ORTHOGONAL_INIT = True
    
    NUM_WORKERS = 8
    BATCH_SIZE = 512
    EPOCH_N = 1000
    SAVE_CYCLE = 250
    VERBOSE_CYCLE = 50
    ENCODER_LR = 0.1
    ENCODER_GAMMA = 0.1
    ENCODER_STEP_CYCLE = 1000
    DECODER_LR = ENCODER_LR
    DECODER_GAMMA = ENCODER_GAMMA
    DECODER_STEP_CYCLE = ENCODER_STEP_CYCLE

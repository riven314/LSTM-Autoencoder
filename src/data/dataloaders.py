import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from src.config import Config as config
from src.common.logger import get_logger
from src.data.sampler import LayoutSampler, collate_pad_fn

logger = get_logger(__name__)


def get_dataloaders():
    default_dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'collate_fn': collate_pad_fn,
        'num_workers': 4, 'shuffle': False
    }
    
    dataloaders = []
    for data_type in ['train', 'val', 'test']:
        dataloader_kwargs = deepcopy(default_dataloader_kwargs)
        if data_type == 'train':
            dataloader_kwargs['shuffle'] = True
        dataloaders.append(
            _get_dataloder(data_type, dataloader_kwargs)
        )

    logger.info('Train, Val, Test DataLoaders are set up')
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    return train_dataloader, val_dataloader, test_dataloader


def _get_dataloder(data_type, dataloader_kwargs):
    assert data_type in ['train', 'val', 'test'], f'Wrong value in data_type: {data_type}'
    dataset_dir = os.path.join(config.DATA_DIR, data_type)
    sampler = LayoutSampler(dataset_dir)
    dataloader = DataLoader(sampler, **dataloader_kwargs)
    return dataloader
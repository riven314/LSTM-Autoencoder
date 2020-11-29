import os
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import src.config as config
from src.common.logger import get_logger
from src.data.layout import Layout

logger = get_logger(__name__)


class LayoutSampler(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        all_fns = os.listdir(data_dir)
        self.json_fns = sorted([fn for fn in all_fns if fn.endswith('.json')])
        self.png_fns = sorted([fn for fn in all_fns if fn.endswith('.png')])

        assert len(self.json_fns) == len(self.png_fns)
        logger.info(f'No. of Layouts: {len(self.json_fns)} ({data_dir})')

    def sample_one_layout(self):
        idx = random.sample(range(len(self)), 1)[0]
        return self[idx]

    def __len__(self):
        return len(self.json_fns)

    def __getitem__(self, idx):
        json_basename = self.json_fns[idx]
        png_basename = self.png_fns[idx]
        assert json_basename.split('.')[0] == png_basename.split('.')[0], 'JSON file unmatched with PNG file'

        json_fn = os.path.join(self.data_dir, json_basename)
        png_fn = os.path.join(self.data_dir, png_basename)
        layout = Layout(json_fn, png_fn)
        return layout


def collate_pad_fn(batch):
    batch = sorted(batch, key = lambda layout: len(layout), reverse = True)
    seq_lens = [len(layout) for layout in batch]
    bboxs_batch = [torch.Tensor(layout.normalized_gt_bboxs) for layout in batch]
    bboxs_batch = pad_sequence(bboxs_batch, batch_first = True)
    classes_batch = [layout.gt_classes for layout in batch]
    return bboxs_batch, seq_lens, classes_batch
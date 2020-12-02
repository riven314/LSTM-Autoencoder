"""
TO BE ADDED:
1. IoU as metric
2. save losses as csv + .png plot
3. print out learning rate
4. add a class to do arithmetic average
5. try positional encoding 
"""
import os
import json
import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pack_padded_sequence

from ranger import Ranger

from src.config import Config
from src.common.logger import get_logger
from src.models.autoencoder import LSTMAutoEncoder

logger = get_logger(__name__)


class Learner:
    def __init__(self, autoencoder: LSTMAutoEncoder, train_loader: DataLoader, val_loader: DataLoader, cfg: Config):
        self.net = autoencoder
        self.train_loader = train_loader
        self.val_loader = val_loader

        # get from config object
        output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT_NAME)
        os.makedirs(output_dir, exist_ok = True)
        self.output_dir = output_dir
        self.device = cfg.DEVICE
        self.epoch_n = cfg.EPOCH_N
        self.save_cycle = cfg.SAVE_CYCLE
        self.verbose_cycle = cfg.VERBOSE_CYCLE
        self.encoder_lr = cfg.ENCODER_LR
        self.decoder_lr = cfg.DECODER_LR
        self.encoder_gamma = cfg.ENCODER_GAMMA
        self.decoder_gamma = cfg.DECODER_GAMMA
        self.encoder_step_cycle = cfg.ENCODER_STEP_CYCLE
        self.decoder_step_cycle = cfg.DECODER_STEP_CYCLE

        # set optimizer and scheduler
        self.encoder_optim = Ranger(
            params = filter(
                lambda p: p.requires_grad, self.encoder.parameters()
            ),
            lr = self.encoder_lr
        )
        self.decoder_optim = Ranger(
            params = filter(
                lambda p: p.requires_grad, self.decoder.parameters()
            ),
            lr = self.encoder_lr
        )
        self.encoder_stepper = StepLR(
            self.encoder_optim, 
            step_size = self.encoder_step_cycle,
            gamma = self.encoder_gamma
        )
        self.decoder_stepper = StepLR(
            self.decoder_optim,
            step_size = self.decoder_step_cycle,
            gamma = self.decoder_gamma
        )
        self.loss = nn.MSELoss()

        # for book-keeping
        self.crt_epoch = 0
        self.train_losses = []
        self.val_losses = []

    @property
    def encoder(self):
        return self.net.encoder

    @property
    def decoder(self):
        return self.net.decoder

    @property
    def signature(self):
        return f'[Epoch: {self.crt_epoch}]'

    @property
    def model_path(self):
        model_name = f'lstm_ae_{self.crt_epoch:04}.pth'
        _path = os.path.join(self.output_dir, model_name)
        return _path

    @property
    def csv_path(self):
        csv_name = f'report.csv'
        return os.path.join(self.output_dir, csv_name)

    def train(self):
        logger.info('Start training...')

        self.net.to(self.device)

        for epoch_i in range(self.epoch_n):
            self.crt_epoch = epoch_i + 1
            start_t = time.time()

            epoch_min = self._train_one_epoch()
            if self.crt_epoch % self.verbose_cycle == 0:
                train_avg_rmse = self.train_losses[-1]
                logger.info(f'{self.signature}:: Train complete. Avg RMSE: {train_avg_rmse:04f}. Time: {epoch_min:03f} mins')

            epoch_min = self._val_one_epoch()
            if self.crt_epoch % self.verbose_cycle == 0:
                val_avg_rmse = self.val_losses[-1]
                logger.info(f'{self.signature}:: Val complete. Avg RMSE: {val_avg_rmse:04f}. Time: {epoch_min:03f} mins')

            total_epoch_min = (time.time() - start_t) / 60.
            if self.crt_epoch % self.verbose_cycle == 0:
                logger.info(f'{self.signature}:: Completed Train + Val. Time: {total_epoch_min} mins')
            
            if self.crt_epoch % self.save_cycle == 0:
                self.save_model()
                logger.info(f'{self.signature}:: Model saved: {self.model_path}')

        logger.info(f'{self.signature}:: Training complete!')
        self.save_model()
        self.save_report()
        logger.info(f'{self.signature}:: Final Model and Report saved: {self.model_path}, {self.csv_path}')

    def _train_one_epoch(self):
        self.net.train()
        start_t = time.time()
        rmse_ls = []

        for bboxs, seq_lens, classes in tqdm(self.train_loader):
            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()

            bboxs = bboxs.to(self.device)
            preds = self.net(bboxs)

            preds = pack_padded_sequence(
                preds, seq_lens, 
                batch_first = True,
                enforce_sorted = True
            )
            targets = pack_padded_sequence(
                bboxs.clone(), seq_lens, 
                batch_first = True,
                enforce_sorted = True
            )
            loss = self.loss(preds.data, targets.data)
            rmse = torch.sqrt(loss)

            rmse.backward()
            self.encoder_optim.step()
            self.decoder_optim.step()
            self.encoder_stepper.step()
            self.decoder_stepper.step()

            report_rmse = float(rmse.data.cpu().numpy())
            rmse_ls.append(report_rmse)

        epoch_min = (time.time() - start_t) / 60.
        avg_rmse = sum(rmse_ls) / len(rmse_ls)
        self.train_losses.append(avg_rmse)
        return epoch_min

    def _val_one_epoch(self):
        self.net.eval()
        start_t = time.time()
        rmse_ls = []

        with torch.no_grad():
            for bboxs, seq_lens, classes in tqdm(self.val_loader):
                bboxs = bboxs.to(self.device)
                preds = self.net(bboxs)

                preds = pack_padded_sequence(
                    preds, seq_lens, 
                    batch_first = True,
                    enforce_sorted = True
                )
                targets = pack_padded_sequence(
                    bboxs.clone(), seq_lens, 
                    batch_first = True,
                    enforce_sorted = True
                )
                loss = self.loss(preds.data, targets.data)
                rmse = torch.sqrt(loss)

                report_rmse = float(rmse.data.cpu().numpy())
                rmse_ls.append(report_rmse)

        epoch_min = (time.time() - start_t) / 60.
        avg_rmse = sum(rmse_ls) / len(rmse_ls)
        self.val_losses.append(avg_rmse)
        return epoch_min

    def save_model(self):
        torch.save(self.net.state_dict(), self.model_path)

    def load_model(self, ckpt_path):
        assert os.path.isfile(ckpt_path), f'Non-exist ckpt_path: {ckpt_path}'
        self.net.load_state_dict(torch.load(ckpt_path))
        logger.info(f'Model loaded: {ckpt_path}')

    def save_report(self):
        losses_dicts = {
            'train_losses': self.train_losses, 'val_losses': self.val_losses
        }
        df = pd.DataFrame(losses_dicts)
        df.to_csv(self.csv_path, index = False)
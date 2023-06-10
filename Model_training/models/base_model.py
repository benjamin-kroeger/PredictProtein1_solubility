import pytorch_lightning as pl
import random
from argparse import Namespace
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from Model_training.utils.collate_functions import collate_pp,collate_seq,collate_pa
from Model_training.utils.constants import seq_encoding_enum
from Model_training.utils.metrics import compute_metrics


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class BaseModel(pl.LightningModule):
    seq_encoding = seq_encoding_enum.pp

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None):
        super().__init__()
        self.val_outputs = defaultdict(list)

        self.save_hyperparameters(args)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.args = args

    def general_step(self, batch, batch_idx, mode):
        encoded_seqs = batch[0]
        solubility_scores = batch[1]

        predicted_solubility_scores = self.forward(encoded_seqs)

        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(predicted_solubility_scores, solubility_scores)

        if mode == 'val':
            metric_dict = compute_metrics(y_true=solubility_scores, y_pred_prob=F.sigmoid(predicted_solubility_scores))
            metric_dict['val_loss'] = loss
            return metric_dict
        if mode == 'train':
            return {'loss': loss}

    def training_step(self, batch, batch_idx):
        metric_dict = self.general_step(batch=batch, batch_idx=batch_idx, mode='train')
        self.log_dict(metric_dict)

        return metric_dict

    def validation_step(self, batch, batch_idx):
        metric_dict = self.general_step(batch=batch, batch_idx=batch_idx, mode='val')
        for key, value in metric_dict.items():
            self.val_outputs[key].append(value)
        return metric_dict

    def on_validation_epoch_end(self):
        for key, value in self.val_outputs.items():
            self.log(key, torch.tensor(value).mean())

        self.val_outputs = defaultdict(list)

    def test_step(self):
        pass

    def train_dataloader(self):
        if self.seq_encoding == seq_encoding_enum.pp:
            return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pp, worker_init_fn=seed_worker)
        if self.seq_encoding == seq_encoding_enum.pa:
            return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pa, worker_init_fn=seed_worker)
        if self.seq_encoding == seq_encoding_enum.seq:
            return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_seq, worker_init_fn=seed_worker)

    def val_dataloader(self):
        if self.seq_encoding == seq_encoding_enum.pp:
            return torch.utils.data.DataLoader(self.train_set, shuffle=False, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pp, worker_init_fn=seed_worker)
        if self.seq_encoding == seq_encoding_enum.pa:
            return torch.utils.data.DataLoader(self.train_set, shuffle=False, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pa, worker_init_fn=seed_worker)

        if self.seq_encoding == seq_encoding_enum.seq:
            return torch.utils.data.DataLoader(self.train_set, shuffle=False, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_seq, worker_init_fn=seed_worker)

    def test_dataloader(self):
        pass

    def configure_optimizers(self):

        params = self.parameters()

        optim = torch.optim.Adam(params=params, betas=(0.9, 0.999), lr=self.args.lr, weight_decay=self.args.reg)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.opt.gamma)

        return [optim] #, [scheduler]


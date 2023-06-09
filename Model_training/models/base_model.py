from typing import Any

import pytorch_lightning as pl
import random
from argparse import Namespace
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset

from Model_training.utils.collate_functions import collate_pp, collate_seq, collate_pa
from Model_training.utils.constants import seq_encoding_enum
from Model_training.utils.metrics import compute_metrics


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BaseModel(pl.LightningModule):
    seq_encoding = seq_encoding_enum.pp

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None, sampler=None):
        super().__init__()
        self.val_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        self.save_hyperparameters(args)
        self.train_set = train_set
        self.val_set = val_set

        self.test_set = test_set
        self.sampler = sampler

        self.args = args

    def general_step(self, batch, batch_idx, mode):
        encoded_seqs = batch[0]
        solubility_scores = batch[1]

        #pre_mem = torch.cuda.memory_allocated()
        predicted_solubility_logits = self.forward(encoded_seqs)
        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(predicted_solubility_logits, solubility_scores)
        #loss.backward(retain_graph=True)
        #gradient_memory = torch.cuda.memory_allocated() -pre_mem
        #self.log('grad_mem_mb',gradient_memory/1024/1024)


        if mode == 'val':
            metric_dict = {'solubility_scores': solubility_scores, 'predicted_scores': F.sigmoid(predicted_solubility_logits), 'val_loss': loss}
            return metric_dict

        if mode == 'test':
            metric_dict = {'solubility_scores': solubility_scores, 'predicted_scores': F.sigmoid(predicted_solubility_logits), 'test_loss': loss}
            return metric_dict
        if mode == 'train':
            return {'loss': loss}

    # def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
    #     super().backward(loss)
    #     gradients = []
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             gradients.append(param.grad)
    #             # For parameters that participate in computations generating gradients,
    #             # also consider the gradients produced by those operations.
    #             if param.grad_fn is not None and param.grad_fn.next_functions:
    #                 for next_fn in param.grad_fn.next_functions:
    #                     if next_fn[0] is not None and hasattr(next_fn[0], 'variable'):
    #                         gradients.append(next_fn[0].variable.grad)
    #     gradient_memory = sum([grad.element_size() * grad.nelement() for grad in gradients])
    #     self.log('grad_mem_mb',gradient_memory/1024/1024)


    def training_step(self, batch, batch_idx):
        metric_dict = self.general_step(batch=batch, batch_idx=batch_idx, mode='train')
        self.log_dict(metric_dict)

        return metric_dict

    def on_train_epoch_start(self):

        tainable_params = torch.tensor(sum(p.numel() for p in self.parameters() if p.requires_grad),dtype=torch.float32)
        self.log('trainable_parameters',tainable_params)

    def validation_step(self, batch, batch_idx):
        metric_dict = self.general_step(batch=batch, batch_idx=batch_idx, mode='val')
        for key, value in metric_dict.items():
            if key == 'val_loss':
                self.val_outputs[key].append(value.cpu().item())
                continue
            self.val_outputs[key].extend(list(torch.squeeze(value.cpu()).numpy()))

    def on_validation_epoch_end(self):

        metric_dict = compute_metrics(y_true=self.val_outputs['solubility_scores'],y_pred_prob=self.val_outputs['predicted_scores'])
        metric_dict['val_loss'] = torch.tensor(self.val_outputs['val_loss']).mean()
        self.log_dict(metric_dict)
        self.val_outputs = defaultdict(list)

    def test_step(self, batch, batch_idx):
        metric_dict = self.general_step(batch=batch, batch_idx=batch_idx, mode='test')
        for key, value in metric_dict.items():
            if key == 'test_loss':
                self.test_outputs[key].append(value.cpu().item())
                continue
            self.test_outputs[key].extend(list(torch.squeeze(value.cpu()).numpy()))

    def on_test_epoch_end(self) -> None:
        metric_dict = compute_metrics(y_true=self.test_outputs['solubility_scores'], y_pred_prob=self.test_outputs['predicted_scores'])
        metric_dict['test_loss'] = torch.tensor(self.val_outputs['test_loss']).mean()
        with open(f'{self.args.model}_test_conf.txt','a') as f:
            y_true = np.array(self.test_outputs['solubility_scores'])
            y_pred = (np.array(self.test_outputs['predicted_scores']) > 0.5).astype(int)

            f.write(str(sklearn.metrics.confusion_matrix(y_true,y_pred)))
            f.write('\n')
        self.log_dict(metric_dict)
        self.test_outputs = defaultdict(list)




    def train_dataloader(self):
        if self.seq_encoding == seq_encoding_enum.pp:
            return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pp, worker_init_fn=seed_worker, drop_last=True)
        if self.seq_encoding == seq_encoding_enum.pa:
            return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pa, worker_init_fn=seed_worker, drop_last=True)
        if self.seq_encoding == seq_encoding_enum.seq:
            return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_seq, worker_init_fn=seed_worker, drop_last=True)

    def val_dataloader(self):
        if self.seq_encoding == seq_encoding_enum.pp:
            return torch.utils.data.DataLoader(self.val_set, shuffle=False, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pp, worker_init_fn=seed_worker)
        if self.seq_encoding == seq_encoding_enum.pa:
            return torch.utils.data.DataLoader(self.val_set, shuffle=False, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pa, worker_init_fn=seed_worker)

        if self.seq_encoding == seq_encoding_enum.seq:
            return torch.utils.data.DataLoader(self.val_set, shuffle=False, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_seq, worker_init_fn=seed_worker)

    def test_dataloader(self):
        if self.seq_encoding == seq_encoding_enum.pp:
            return torch.utils.data.DataLoader(self.test_set, shuffle=False, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pp, worker_init_fn=seed_worker,
                                               sampler=self.sampler)
        if self.seq_encoding == seq_encoding_enum.pa:
            return torch.utils.data.DataLoader(self.test_set, shuffle=False, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_pa, worker_init_fn=seed_worker,
                                               sampler=self.sampler)
        if self.seq_encoding == seq_encoding_enum.seq:
            return torch.utils.data.DataLoader(self.test_set, shuffle=False, batch_size=self.args.batch_size, pin_memory=True,
                                               num_workers=self.args.num_workers, collate_fn=collate_seq, worker_init_fn=seed_worker,
                                               sampler=self.sampler)

    def configure_optimizers(self):

        params = self.parameters()

        optim = torch.optim.Adam(params=params, betas=(0.9, 0.999), lr=self.args.lr, weight_decay=self.args.reg)


        return [optim]

# TODO: create base model
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset
from Model_training.utils.constants import seq_encoding_enum
from Model_training.utils.metrics import compute_metrics
import torch.nn.functional as F


class BaseModel(pl.LightningModule):
    seq_encoding = seq_encoding_enum.pp

    def __init__(self, args, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None):
        super().__init__()
        self.save_hyperparameters(args)

        self.val_outputs = []

    def _general_step(self, batch, batch_idx, mode):
        encoded_seqs = batch[0]
        solubility_scores = batch[1]

        predicted_solubility_scores = self.forward(encoded_seqs)

        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(predicted_solubility_scores, solubility_scores)

        if mode == 'val':
            metric_dict = compute_metrics(y_true=solubility_scores, y_pred_prob=F.sigmoid(predicted_solubility_scores))
            metric_dict['val_loss'] = loss
        if mode == 'train':
            return {'train_loss': loss}

    def training_step(self, batch, batch_idx):
        metric_dict = self._general_step(batch=batch, batch_idx=batch_idx, mode='train')
        self.log_dict(metric_dict)

        return metric_dict

    def validation_step(self,batch,batch_idx):
        metric_dict = self._general_step(batch=batch, batch_idx=batch_idx, mode='val')
        self.val_outputs.append(metric_dict)
        self.log_dict(metric_dict)

        return metric_dict

    def test_step(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def configure_optimizers(self):
        pass

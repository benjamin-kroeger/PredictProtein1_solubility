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

from .base_model import BaseModel
from utils.collate_functions import collate_pp, collate_seq, collate_pa
from utils.constants import seq_encoding_enum
from utils.metrics import compute_metrics

class LightAttention(BaseModel):
    seq_encoding = seq_encoding_enum.pa

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None,sampler=None,embeddings_dim=1024, output_dim=1, dropout=0.25, kernel_size=9, conv_dropout = 0.25):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set,sampler=sampler)

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, np.float16(-np.inf))

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]

    def general_step(self, batch, batch_idx, mode):
        encoded_seqs = self.pad_sequence(batch[0], 1024)
        masks = encoded_seqs.bool().any(axis=2)
        encoded_seqs = encoded_seqs.permute(0,2,1)
        solubility_scores = batch[1]

        predicted_solubility_scores = self.forward(encoded_seqs, masks)

        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(predicted_solubility_scores, solubility_scores)

        if mode == 'val':
            metric_dict = compute_metrics(y_true=solubility_scores, y_pred_prob=F.sigmoid(predicted_solubility_scores))
            metric_dict['val_loss'] = loss
            return metric_dict

        if mode == 'test':
            metric_dict = compute_metrics(y_true=solubility_scores, y_pred_prob=F.sigmoid(predicted_solubility_scores))
            metric_dict['test_loss'] = loss
            return metric_dict
        if mode == 'train':
            return {'loss': loss}

    def pad_sequence(self, sequence, max_length):
        # Get the original sequence length
        original_length = sequence.size(1)
        
        # If the original length is already equal to or greater than the maximum length, return the sequence as is
        if original_length >= max_length:
            return sequence
        
        # Calculate the number of elements to be padded
        pad_length = max_length - original_length
        
        # Create a padding tensor of zeros with the appropriate shape
        padding = torch.zeros(sequence.size(0), pad_length, sequence.size(2)).to(sequence.device)
        
        # Concatenate the padding tensor to the original sequence along the second dimension
        padded_sequence = torch.cat([sequence, padding], dim=1)
        
        return padded_sequence




class LightAttention1(BaseModel):
    seq_encoding = seq_encoding_enum.pa

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None, sampler=None, embeddings_dim=1024, output_dim=1, dropout=0.25, kernel_size=9, conv_dropout=0.25):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set, sampler=sampler)

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)  # Added BatchNorm1d layer
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, np.float16(-np.inf))

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]

    def general_step(self, batch, batch_idx, mode):
        encoded_seqs = self.pad_sequence(batch[0], 1024)
        masks = encoded_seqs.bool().any(axis=2)
        encoded_seqs = encoded_seqs.permute(0,2,1)
        solubility_scores = batch[1]

        predicted_solubility_scores = self.forward(encoded_seqs, masks)

        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(predicted_solubility_scores, solubility_scores)

        if mode == 'val':
            metric_dict = compute_metrics(y_true=solubility_scores, y_pred_prob=F.sigmoid(predicted_solubility_scores))
            metric_dict['val_loss'] = loss
            return metric_dict

        if mode == 'test':
            metric_dict = compute_metrics(y_true=solubility_scores, y_pred_prob=F.sigmoid(predicted_solubility_scores))
            metric_dict['test_loss'] = loss
            return metric_dict
        if mode == 'train':
            return {'loss': loss}

    def pad_sequence(self, sequence, max_length):
        # Get the original sequence length
        original_length = sequence.size(1)
        
        # If the original length is already equal to or greater than the maximum length, return the sequence as is
        if original_length >= max_length:
            return sequence
        
        # Calculate the number of elements to be padded
        pad_length = max_length - original_length
        
        # Create a padding tensor of zeros with the appropriate shape
        padding = torch.zeros(sequence.size(0), pad_length, sequence.size(2)).to(sequence.device)
        
        # Concatenate the padding tensor to the original sequence along the second dimension
        padded_sequence = torch.cat([sequence, padding], dim=1)
        
        return padded_sequence

from .base_model import BaseModel
from torch import nn
from argparse import Namespace
from torch.utils.data import Dataset

from Model_training.utils.constants import seq_encoding_enum


class test_model(BaseModel):
    seq_encoding = seq_encoding_enum.pp

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set)

        self.model = nn.Linear(in_features=1024, out_features=1)

    def forward(self, pp_embedd):
        return self.model(pp_embedd)

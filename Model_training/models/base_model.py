# TODO: create base model
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset
from Model_training.utils.constants import seq_encoding_enum

class BaseModel(pl.LightningModule):
    seq_encoding = seq_encoding_enum.pp
    def __init__(self,args,train_set:Dataset=None,val_set:Dataset=None,test_set:Dataset=None):


        super().__init__()
       
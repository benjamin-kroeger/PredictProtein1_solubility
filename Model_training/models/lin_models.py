from .base_model import BaseModel
from torch import nn
from argparse import Namespace
from torch.utils.data import Dataset

from Model_training.utils.constants import seq_encoding_enum


class test_model(BaseModel):
    seq_encoding = seq_encoding_enum.pp

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None,sampler=None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set,sampler=sampler)

        self.model = nn.Linear(in_features=1024, out_features=1)

    def forward(self, pp_embedd):
        return self.model(pp_embedd)


class newLinearModel(BaseModel):

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set)
        self.fc1 = nn.Linear(1024, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)

    def forward(self, pp_embedds):
        out = self.fc1(pp_embedds)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class TwoLayerLin512(BaseModel):

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None,
                 test_set: Dataset = None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set)
        self.fc1 = nn.Linear(1024, 512)
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 1)

    def forward(self, pp_embedds):
        out = self.fc1(pp_embedds)
        out = self.leakyrelu(out)
        out = self.fc2(out)
        return out


class TwoLayerLin256(BaseModel):

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None,
                 test_set: Dataset = None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set)
        self.fc1 = nn.Linear(1024, 256)
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, pp_embedds):
        out = self.fc1(pp_embedds)
        out = self.leakyrelu(out)
        out = self.fc2(out)
        return out


class TwoLayerLin128(BaseModel):

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None,
                 test_set: Dataset = None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set)
        self.fc1 = nn.Linear(1024, 128)
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, pp_embedds):
        out = self.fc1(pp_embedds)
        out = self.leakyrelu(out)
        out = self.fc2(out)
        return out


class TwoLayerLin64(BaseModel):

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None,
                 test_set: Dataset = None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set)
        self.fc1 = nn.Linear(1024, 64)
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, pp_embedds):
        out = self.fc1(pp_embedds)
        out = self.leakyrelu(out)
        out = self.fc2(out)
        return out

class OneLayerLin(BaseModel):

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None,
                 test_set: Dataset = None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set)
        self.model = nn.Linear(1024, 1)

    def forward(self, pp_embedds):
        return self.model(pp_embedds)


class ThreeLayerLin128x64(BaseModel):

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None,
                 test_set: Dataset = None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set)
        self.fc1 = nn.Linear(1024, 128)
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 64)
        self.leakyrelu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, pp_embedds):
        out = self.fc1(pp_embedds)
        out = self.leakyrelu(out)
        out = self.fc2(out)
        out = self.leakyrelu2(out)
        out = self.fc3(out)
        return out


class Two_lin_layer(BaseModel):
    seq_encoding = seq_encoding_enum.pp

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None, sampler=None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set, sampler=sampler)

        self.model = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                   nn.BatchNorm1d(256),
                                   nn.Linear(in_features=256,out_features=1))

    def forward(self, pp_embedd):
        return self.model(pp_embedd)


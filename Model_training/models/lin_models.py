import torch

from .base_model import BaseModel
from torch import nn
from argparse import Namespace
from torch.utils.data import Dataset

from Model_training.utils.constants import seq_encoding_enum


class test_model(BaseModel):
    seq_encoding = seq_encoding_enum.pp

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None, sampler=None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set, sampler=sampler)

        self.model = nn.Linear(in_features=1024, out_features=1)

    def forward(self, pp_embedd):
        return self.model(pp_embedd)


class fail_model(BaseModel):
    seq_encoding = seq_encoding_enum.pp

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None, sampler=None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set, sampler=sampler)

        self.model = nn.Linear(in_features=1024, out_features=1)

        self.lr_schedule = {
            0: 1,  # Learning rate for epoch 0
            3: 0.1,  # Learning rate for epoch 3
            4: 0.2,  # Learning rate for epoch 4
            5: 0.3,  # Learning rate for epoch 5
            6: 0.3,
            7: 0.3,
            8: 0.4,
            9: 0.5,
            10: 0.6,
            11: 0.7,
            12: 0.8,
            13: 0.9
        }

    def forward(self, pp_embedd):

        return self.model(pp_embedd)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self):

        params = self.parameters()

        optim = torch.optim.Adam(params=params, betas=(0.9, 0.999), lr=self.args.lr, weight_decay=self.args.reg)

        def lr_lambda(epoch):
            if epoch in self.lr_schedule:
                return self.lr_schedule[epoch]
            else:
                return 1

        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

        return [optim],[scheduler]

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
                                   nn.Linear(in_features=256, out_features=1))

    def forward(self, pp_embedd):
        return self.model(pp_embedd)


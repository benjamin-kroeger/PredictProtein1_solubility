# This file is used to define custom collate functions eg for per residue embeddings
import torch
# TODO: implement a collate function for per residue embeddings

# TODO implemt colalte for stacking emebdds

from torch import nn


def my_collate(batch):
    batch_unzipped = list(zip(*batch))
    embedds = torch.stack(batch_unzipped[0])
    sol = torch.unsqueeze(torch.tensor(batch_unzipped[1]),1)

    return embedds, sol
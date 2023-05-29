import torch
from torch.utils.data import Dataset
from constants import seq_encoding_enum, mode_enum
import h5py
from tqdm import tqdm
import numpy as np


class NetsolpDataset(Dataset):

    # seq_encoding param is infered from the model
    def __init__(self, seq_encoding: seq_encoding_enum, set_mode: mode_enum, val_partion: int, dtype: torch.dtype, path_to_seq_data: str,
                 path_to_embedds: str = None):
        assert 0 <= val_partion <= 4, "The selected partiontion is unknown"
        if seq_encoding != seq_encoding.seq and path_to_embedds is None:
            raise ValueError('path_to_embedds must be defined when trying to use embeddings')

        # load all files

        # keep seq or embedd based in seq_encodiung

        # list of tuples
        # (id,sol,seq/embeddpp/embedpa,part)

        # frop 4 or 1 partions depending on mode val protin
        # id is used to map embedds to sol and part

        # only keep sol,seq/embeddpp/embedpa
        self.data = self._drop_unecesary_partition(data=None, mode=set_mode, val_parition=val_partion)

    # TODO: Combine embedd tupels with data

    def _read_embeddings_from_h5(self, path_to_embedds: str, dtype: torch.dtype) -> list[tuple[int, torch.tensor]]:
        """
        Reads any embedding type from a h5 file as long as every embed is stored as its own dataset.
        It does not distinguish between per residue and per amino embeddings
        :param path_to_embedds: path to the h5 file
        :param dtype: The torch dtype the tensors shall be returned as e.g float16/32/64
        :return: A list of tuples with the id and the embedd tensor
        """
        embeddings = []
        embedd_file = h5py.File(path_to_embedds)
        pbar_desc = f'Loading embeddings'
        with tqdm(total=len(embedd_file.keys()), desc=pbar_desc) as pbar:
            for key in embedd_file.keys():
                embedding = np.array(embedd_file[key])
                embedding = torch.tensor(embedding).to(dtype)
                embeddings.append((key, embedding))
                pbar.update(1)

        return embeddings

    def _drop_unecesary_partition(self, data: list[tuple], mode: mode_enum, val_parition: int) -> list[tuple]:
        """
        Drops all entries that are not allowed in the dataset depening on whether this Dataset is the train or validation set and on which partition
        is meant to be the validation partition
        :param data: A list of tulples with the
        :param mode: train or validation mode
        :param val_parition: which partition shall be used for validation
        :return: A list of tuples
        """

        if mode == mode_enum.val:
            # only keep entries that are part of the validation partition
            filtered_data = [x for x in data if x[3] == val_parition]
        elif mode == mode_enum.train:
            # only keep entries that are not part of the validation partion
            filtered_data = [x for x in data if x[3] != val_parition]

        return filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.data[index]

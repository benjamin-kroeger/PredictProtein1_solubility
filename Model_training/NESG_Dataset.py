from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.constants import seq_encoding_enum, mode_enum


class NESGDataset(Dataset):

    # seq_encoding param is infered from the model
    def __init__(self, seq_encoding: seq_encoding_enum, dtype: torch.dtype,
                 path_to_seq_data: str,
                 path_to_embedds: str = None):

        if seq_encoding != seq_encoding.seq and path_to_embedds is None:
            raise ValueError('path_to_embedds must be defined when trying to use embeddings')

        # load all files
        if seq_encoding == seq_encoding.seq:
            solubility_data = self._read_csv(path_to_seq_data).values.tolist()
            solubility_data = [(data_tuple[0], data_tuple[2], torch.tensor(data_tuple[1]).to(dtype)) for data_tuple in solubility_data]
        else:
            seq_data = self._read_csv(path_to_seq_data)
            embedding_data = self._read_embeddings_from_h5(path_to_embedds, dtype=dtype)
            solubility_data = self._merge_dataset(embedding_data, seq_data,dtype=dtype)

        self.data = self._drop_unnecessary(solubility_data)

    def _read_csv(self, path: str):
        data = Path(path)
        return pd.read_csv(data)

    def _merge_dataset(self, embeddings: list[tuple[str, torch.tensor]], csv: pd.DataFrame,dtype:torch.dtype) -> list[tuple]:
        """
        Merge data from solubility trainset (csv) with embeddings read from h5 file
        :param embeddings: protein embeddings read from h5, result from _read_embeddings_from_h5
        :param csv: solubility trainset read from csv, result from _read_csv
        :return: list of tuples, each tuple: ( id, embedding-tensor, fasta-seq, solubility, partition )
        """
        dataset = []
        for emb in embeddings:
            if emb[0] in csv["sid"].tolist():
                _, sol, fasta, partition = csv[csv["sid"] == emb[0]].iloc[0]
                dataset.append((emb[0], emb[1], torch.tensor(sol).to(dtype), partition))
        return dataset

    def _drop_unnecessary(self, dataset: list[tuple[int, torch.tensor, float, str, float]]) -> list[tuple]:
        """
        Remove everything from data except for fasta-seq and solubility score
        :param dataset: whole dataset, result from _merge_dataset function
        :return list of tuples, each tuple: (fasta-seq, solubility)
        """
        clean_dataset = []
        for elem in dataset:
            clean_dataset.append((elem[1], elem[2]))
        return clean_dataset

    def _read_embeddings_from_h5(self, path_to_embedds: str, dtype: torch.dtype) -> list[tuple[str, torch.tensor]]:
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

    # TODO: rewrite this so that we don't have to reload all our data for each split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]




if __name__ == "__main__":
    d = NESGDataset(seq_encoding_enum.seq, torch.float32,
                       "/home/benjaminkroeger/Documents/Master/Master_2_Semester/Predictprotein2/predictprotein1_solubility/Data/NESG_testset_formatted.csv",
                       "/home/benjaminkroeger/Documents/Master/Master_2_Semester/Predictprotein2/predictprotein1_solubility/Data/test_embedds_pp.h5")

    print(d.__getitem__(1))
    print(len(d))

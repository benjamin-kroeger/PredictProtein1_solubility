import torch
from torch.utils.data import Dataset
from constants import seq_encoding_enum
class NetsolpDataset(Dataset):

    # seq_encoding param is infered from the model
    def __init__(self,seq_encoding:seq_encoding_enum,set_mode:str,val_partion:int,dtype:torch.dtype):

        # load all files

        # keep seq or embedd based in seq_encodiung

        #list of tuples
        # (id,sol,seq/embeddpp/embedpa,part)

        # frop 4 or 1 partions depending on mode val protin
        self.data = []
    def read_embeddings_from_h5(self,path_to_h5:str) -> dict :
        pass

















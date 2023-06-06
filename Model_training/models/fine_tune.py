from .base_model import BaseModel
from torch import nn
from argparse import Namespace
from torch.utils.data import Dataset
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
from Model_training.utils.constants import seq_encoding_enum

class fine_tune_t5(BaseModel):
    seq_encoding = seq_encoding_enum.seq
    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set)

        self.model = nn.Linear(in_features=1024, out_features=1)
        self.dropout = nn.Dropout(args.drop)
        model_name = r'Rostlab/prot_t5_xl_uniref50'
        self.plm_model = T5EncoderModel.from_pretrained(model_name)
        # freeze the grad
        self.plm_model.requires_grad_(False)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

        # imnplement a stepped learning rate
        # implement batch norm and
        # implement grad clipping
    def forward(self, sequences):
        # unfreeze layers based on epoch
        if self.current_epoch == 1:
            params = [x for x in self.plm_model.parameters()]
            params[-1].requires_grad()

        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        embedding_repr = self.plm_model(input_ids=input_ids, attention_mask=attention_mask)

        return_tensor = []
        for i in range(len(sequences)):
            return_tensor.append(embedding_repr.last_hidden_state[i, :sum(attention_mask[i])].mean(axis=0))

        pa_embedds = torch.stack(return_tensor)

        droped = self.dropout(pa_embedds)
        output = self.model(droped)

        return output

    def configure_optimizers(self):

        params = self.parameters()

        optim = torch.optim.Adam(params=params, betas=(0.9, 0.999), lr=self.args.lr, weight_decay=self.args.reg)

        return [optim]
from .base_model import BaseModel
from torch import nn
from argparse import Namespace
from torch.utils.data import Dataset
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
from Model_training.utils.constants import seq_encoding_enum
from .LoRA import LoRALinear,modify_with_lora


class fine_tune_t5(BaseModel):
    seq_encoding = seq_encoding_enum.seq


    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None, sampler=None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set, sampler=sampler)

        self.model = nn.Linear(in_features=1024, out_features=1)
        model_name = r'Rostlab/prot_t5_xl_uniref50'
        self.plm_model = T5EncoderModel.from_pretrained(model_name)
        # freeze the grad
        self.plm_model.requires_grad_(False)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

        self.lr_schedule = {
            0: args.lr,  # Learning rate for epoch 0
            3: 0.0001,  # Learning rate for epoch 3
            4: 0.0002,  # Learning rate for epoch 4
            5: 0.0003,  # Learning rate for epoch 5
            6: 0.0003,
            7: 0.0003,
            8: 0.0004,
            9: 0.0005,
            10: 0.0006,
            11: 0.0007,
            12: 0.0008,
            13: 0.0009,
            14: 0.001
        }
        self.unfroze1 = False
        self.unfroze2 = False
        self.unfroze3 = False

        # imnplement a stepped learning rate
        # implement batch norm and
        # implement grad clipping

    def training_step(self, batch, batch_idx):
        # unfreeze layers based on epoch
        if self.current_epoch == 3 and not self.unfroze1:
            named_module_dict = dict(self.plm_model.named_modules())
            named_module_dict['encoder.block.23.layer.0.SelfAttention.q'].weight.requires_grad_(True)
            named_module_dict['encoder.block.23.layer.0.SelfAttention.k'].weight.requires_grad_(True)
            named_module_dict['encoder.block.23.layer.0.SelfAttention.v'].weight.requires_grad_(True)
            named_module_dict['encoder.block.23.layer.0.SelfAttention.o'].weight.requires_grad_(True)

        if self.current_epoch == 4 and not self.unfroze1:
            named_module_dict = dict(self.plm_model.named_modules())
            named_module_dict['encoder.block.22.layer.0.SelfAttention.q'].weight.requires_grad_(True)
            named_module_dict['encoder.block.22.layer.0.SelfAttention.k'].weight.requires_grad_(True)
            named_module_dict['encoder.block.22.layer.0.SelfAttention.v'].weight.requires_grad_(True)
            named_module_dict['encoder.block.22.layer.0.SelfAttention.o'].weight.requires_grad_(True)


        metric_dict = self.general_step(batch=batch, batch_idx=batch_idx, mode='train')
        self.log_dict(metric_dict)

        return metric_dict

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def forward(self, sequences):

        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        embedding_repr = self.plm_model(input_ids=input_ids, attention_mask=attention_mask)

        return_tensor = []
        for i in range(len(sequences)):
            return_tensor.append(embedding_repr.last_hidden_state[i, :sum(attention_mask[i])].mean(axis=0))

        pa_embedds = torch.stack(return_tensor)

        output = self.model(pa_embedds)

        return output


    def configure_optimizers(self):

        params = self.parameters()

        optim = torch.optim.Adam(params=params, betas=(0.9, 0.999), lr=self.args.lr, weight_decay=self.args.reg)

        def lr_lambda(epoch):
            if epoch in self.lr_schedule:
                return self.lr_schedule[epoch]
            else:
                return self.args.lr

        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]


class LoRAConfig:
    def __init__(self):
        self.lora_rank = 1
        self.lora_init_scale = 0.01
        self.lora_modules = ".*SelfAttention|.*EncDecAttention"
        self.lora_layers = "q|k|v|o"
        self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
        self.lora_scaling_rank = 1


class fine_tune_lora(BaseModel):
    seq_encoding = seq_encoding_enum.seq

    def __init__(self, args: Namespace, train_set: Dataset = None, val_set: Dataset = None, test_set: Dataset = None, sampler=None):
        super().__init__(args=args, train_set=train_set, val_set=val_set, test_set=test_set, sampler=sampler)

        self.final_linear = nn.Linear(in_features=1024, out_features=1)

        config = LoRAConfig()
        model_name = r'Rostlab/prot_t5_xl_uniref50'
        self.plm_model = T5EncoderModel.from_pretrained(model_name).requires_grad_(False)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

        self.plm_model = modify_with_lora(self.plm_model,config)


    def forward(self, sequences):

        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        embedding_repr = self.plm_model(input_ids=input_ids, attention_mask=attention_mask)

        return_tensor = []
        for i in range(len(sequences)):
            return_tensor.append(embedding_repr.last_hidden_state[i, :sum(attention_mask[i])].mean(axis=0))

        pa_embedds = torch.stack(return_tensor)

        output = self.final_linear(pa_embedds)

        return output



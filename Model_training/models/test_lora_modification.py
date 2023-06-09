import torch
import torch.nn as nn
import torch.nn.functional as F
import re


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, scaling_rank, init_scale):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
            if init_scale < 0:
                self.lora_b = nn.Parameter(torch.randn(linear_layer.out_features, rank) * init_scale)
            else:
                self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank) * init_scale
                )
            else:
                self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features, self.scaling_rank))

    def forward(self, input):
        if self.scaling_rank == 1 and self.rank == 0:
            # parsimonious implementation for ia3 and lora scaling
            if self.multi_lora_a.requires_grad:
                hidden = F.linear((input * self.multi_lora_a.flatten()), self.weight, self.bias)
            else:
                hidden = F.linear(input, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten()
            return hidden
        else:
            # general implementation for lora (adding and scaling)
            weight = self.weight
            if self.scaling_rank:
                weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
            if self.rank:
                weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
            return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, rank={}, scaling_rank={}".format(
            self.in_features, self.out_features, self.bias is not None, self.rank, self.scaling_rank
        )


def modify_with_lora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(layer, config.lora_rank, config.lora_scaling_rank, config.lora_init_scale),
                    )
    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5EncoderModel, T5Tokenizer


    class LoRAConfig:
        def __init__(self):
            self.lora_rank = 4
            self.lora_init_scale = 0.01
            self.lora_modules = ".*SelfAttention|.*EncDecAttention"
            self.lora_layers = "q|k|v|o"
            self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
            self.lora_scaling_rank = 1
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

    config = LoRAConfig()
    model_name = r'Rostlab/prot_t5_xl_uniref50'
    model = T5EncoderModel.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

    input_seq = tokenizer(
        ["Applies a linear transformation to the incoming data."],
        return_tensors="pt",
    )
    target_seq = tokenizer(
        ["Parameters: in_features - size of each input sample. out_features - size of each output sample."],
        return_tensors="pt",
    )

    print("Old model")
    print(model)

    model = modify_with_lora(model, config)

    print("New model")
    print(model)
    sequences = ['MWTYQKADNPVFQYKAETTFDVPLERAVAVILDVDRAAQWVPYMGKVQMLSQDEKKGEFTLYMVLDFPFPLKD']
    device = torch.device('cuda')
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    model = model.to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        


    print("Trainable parameters")
    print(
        [
            p_name
            for p_name in dict(model.named_parameters()).keys()
            if re.fullmatch(config.trainable_param_names, p_name)
        ]
    )

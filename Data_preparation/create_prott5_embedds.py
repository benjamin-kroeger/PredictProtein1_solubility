import pandas as pd
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
import h5py
from tqdm import tqdm
from pathlib import Path
import re

sol_df = pd.read_csv(r'/home/benjaminkroeger/Documents/Master/Master_2_Semester/Predictprotein2/predictprotein1_solubility/Data/NESG_testset_formatted.csv')

model_name = r'Rostlab/prot_t5_xl_uniref50'
# load the model in half precision
model = T5EncoderModel.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name ,do_lower_case=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def create_embeddings(sequences:list,tokenizer : T5Tokenizer,model:T5EncoderModel,device:torch.device)->torch.tensor:

    # clean seqs
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    model = model.to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)


    return_tensor= []
    for i in range(len(sequences)):
        return_tensor.append(embedding_repr.last_hidden_state[i,:sum(attention_mask[i])])

    return torch.stack(return_tensor)[0]

def store_in_h5(header,embedding_tensors,filename):

    if Path(filename).is_file():
        #print("Appending to file...")
        h5f = h5py.File(filename, 'a')
    else:
        #print("Making file...")
        h5f = h5py.File(filename, 'w')


    h5f.create_dataset(header, data=embedding_tensors)
    h5f.close()
    #print("Finished")

ids = sol_df['sid'].tolist()
seqs = sol_df['fasta'].tolist()
output_h5_file_pp = '../Data/test_embedds_pp.h5'
output_h5_file_pa = '../Data/test_embedds_pa.h5'

for id,fasta in tqdm(zip(ids,seqs),total=len(ids)):
    tensor = create_embeddings(sequences=[fasta],tokenizer=tokenizer,model=model,device=device)
    store_in_h5(header=id,embedding_tensors=tensor.cpu(),filename=output_h5_file_pa)
    store_in_h5(header=id,embedding_tensors=tensor.cpu().mean(axis=0),filename=output_h5_file_pp)
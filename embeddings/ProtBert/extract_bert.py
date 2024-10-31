from transformers import BertModel, BertTokenizer
import re
import torch
from tqdm import tqdm

fasta_file = './all_mdcath_sequences.fasta'


tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")

with open(fasta_file, 'r') as f:
    fasta = f.readlines()
for sequence in tqdm(fasta):
    if sequence.startswith(">"):
        sysname = sequence.strip().strip(">")
        #print(sysname)
    else:
        sequence_extracted = ' '.join(list(sequence.strip()))
        sequence_extracted = re.sub(r"[UZOB]", "X", sequence_extracted)
        encoded_input = tokenizer(sequence_extracted, return_tensors='pt')
        output = model(**encoded_input)
        #print(output.pooler_output.shape)
        #print(output.last_hidden_state.shape)
        embedding = output.last_hidden_state
        torch.save(embedding, f'./embeddings/{sysname}_prot_bert.pt')

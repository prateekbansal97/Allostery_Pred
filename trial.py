import torch
import torch.nn.functional as F
import numpy as np

def rel_send_rec(num_residues):
    off_diag = np.ones([num_residues, num_residues]) - np.eye(num_residues)
    columns = torch.LongTensor(np.where(off_diag)[1])
    rows = torch.LongTensor(np.where(off_diag)[0])
    rel_rec = F.one_hot(columns).float()
    rel_send = F.one_hot(rows).float()
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)
    return rel_rec, rel_send

p, q = rel_send_rec(99)
print(p.shape, q.shape)

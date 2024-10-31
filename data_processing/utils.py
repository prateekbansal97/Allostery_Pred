import numpy as np
import torch
import torch.nn.functional as F


def load_windows(timestep):
    return {
        'train': [0, int(timestep * 0.8)],
        'valid': [int(timestep * 0.8), int(timestep * 0.9) + 1],
        'test': [int(timestep * 0.9) + 1, int(timestep) - 1]
    }

def check_sizes(dataset):
    sizes = np.array([j.shape for j in dataset])
    all_equal = np.all(sizes == sizes[0])
    return all_equal, sizes

def rel_send_rec(num_residues):
    off_diag = np.ones([num_residues, num_residues]) - np.eye(num_residues)
    columns = torch.LongTensor(np.where(off_diag)[1])
    rows = torch.LongTensor(np.where(off_diag)[0])
    rel_rec = F.one_hot(columns).float()
    rel_send = F.one_hot(rows).float()
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)
    return rel_rec, rel_send


def n_res_dict(sysname):
    resdict =  {
        'agtr2_5UNG':335,
		'cnr2_5ZTY':320,
		'c5ar1_6C1R':328,
		'ada1b_7B6W':351,
		'5ht4r_7XT8':330,
		'ada2c_6KUW':455,
		'p2ry1_4XNV':335,
		'5ht6r_7XTB':339,
		's1pr1_7TD4':324,
		'acm4_5DSG':474,
		'mtr1b_6ME6':329,
		'ox2r_5WQC':384,
		'pe2r2_7CX3':331,
		'acm2_5ZKC':459,
		'cltr2_6RZ7':323,
		'aa2ar_5NM4':305,
		'gpr52_6LI0':339,
		'5ht1e_7E33':359,
		'5ht7r_7XTC':404,
		'oxyr_6TPK':339,
		'bkrb2_7F2O':347,
		'gnrhr_7BR3':329,
		'mc4r_6W25':321,
		'ccr5_5UIW':317,
		'5ht1b_4IAR':388,
		'pe2r4_5YWY':347,
		'oprd_4N6H':339,
		'galr1_7WQ3':321,
		'opsd_5W0P':344,
		'ccr6_6WWZ':337,
		'npy1r_5ZBQ':338,
		'par2_5NDD':360,
		'5ht1d_7E32':372,
		'ccr2_6GPX':309,
		'drd4_5WIU':463,
		'lshr_7FIH':645,
		'5ht1f_7EXD':362,
		'5ht2b_4IB4':401,
		'ntr1_6OS9':376,
		'mrgx4_7S8P':281,
		'fpr2_7WVX':318,
		's1pr2_7T6B':291,
		'adrb2_6PS2':343,
		'adrb1_7BVQ':393,
		'lpar1_4Z35':328,
		'v2r_7DW9':340,
		'mrgx2_7S8L':286,
		'hrh1_3RZE':486,
		'ptafr_5ZKQ':301,
		'5ht5a_7UM5':353,
		'mshr_7F4H':313,
		'galr2_7WQ4':308,
		'gp139_7VUG':305,
		'fpr1_7T6T':317,
		'drd3_7CMV':401,
		'5ht1a_7E2X':416,
		'ox1r_6TOS':378,
		's1pr5_7EW1':311,
		'ccr9_5LWE':341,
		'mc4r_7PIU':317,
		'pe2r3_6M9T':354,
		'5ht5a_7UM4':342,
		's1pr1_3V2Y':327,
		'mtr1a_6ME2':321,
		'npy2r_7DDZ':344,
		's1pr3_7EW3':313,
		'oprx_5DHG':332,
		'agtr1_6OS2':319,
		'lpar1_7TD0':324,
		'gpbar_7CFN':292,
		'ccr1_7VL9':319,
		'bkrb1_7EIB':331,
		'aa1r_5UEN':310,
		'drd3_3PBL':401,
		'apj_5VBL':331,
		'ta2r_6IIU':324,
		'ghsr_7F83':341,
		'agtr1_4ZUD':305,
		'cxcr2_6LFL':337,
		'drd1_7JVP':347,
		'ada2b_6K41':443,
		'lshr_7FIJ':645
    }
    return resdict[sysname]

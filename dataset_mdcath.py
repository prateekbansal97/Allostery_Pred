import pickle
from data_processing.parm_traj_manager import ParmTrajManager
from data_processing.dataset_generator import DatasetGenerator
from data_processing.utils import load_windows, n_res_dict, rel_send_rec
from train import encoder, decoder, train_val
import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
import glob






#Hyperparameters
ndims = 6
n_hidden_encoder = 2 
n_hidden_decoder = 2 
n_edges =  2 
encoder_dropout = 0.05
decoder_dropout = 0.05
skip_first = True
lr = 0.0005
lr_decay = 200 
gamma = 0.5
nepochs = 500
#train_batch_size=batch_size
#valid_batch_size=batch_size

# Hyperparameters
group_dir = './data/groups'
xtcs_list_file = './data/xtcs_list'
strip_parms_list_file = './data/strip_parms_list'
parm_traj_output_path = './data/parm_traj_dict.pkl'

parm_traj_manager = ParmTrajManager(group_dir, xtcs_list_file, strip_parms_list_file)
parm_traj_dict = parm_traj_manager.load_parm_traj_dict(parm_traj_output_path)

dataset_generator = DatasetGenerator(parm_traj_dict, mdcath=True)

dataset_generator.process_all_systems()

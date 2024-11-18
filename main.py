"""
Authored by Prateek Bansal, Nov 2, 2024
University of Illinois Urbana Champaign
pdb3@illinois.edu

Script for training a Neural Relational Inference (NRI) model on molecular dynamics simulations.

This script performs the following:
1. Parses command-line arguments for system name and batch size.
2. Initializes hyperparameters and data processing utilities.
3. Sets up the training and validation data loaders.
4. Initializes the encoder and decoder models for NRI.
5. Loads model checkpoints if available and resumes training.
6. Executes the training loop with logging and saves the training results.

Dependencies:
- PyTorch for deep learning.
- ParmTrajManager, DatasetGenerator, and utility functions for data processing.
- train module for training and validation of the models.

Command-line Arguments:
- `--sysname`: Name of the system for NRI training.
- `--batch-size`: Batch size for training and validation.
"""

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



parser = argparse.ArgumentParser('Neural relational inference for molecular dynamics simulations')
parser.add_argument('--sysname', type=str, required=True, default='',
                    help='Name of the system for NRI.')
parser.add_argument('--batch-size', type=int, default=24,
                    help='Name of the system for NRI.')
args = parser.parse_args()

sysname = args.sysname
batch_size = args.batch_size

print(f"INFO:: Starting training for system {sysname}")

device = torch.device("cuda:0")



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
train_batch_size=batch_size
valid_batch_size=batch_size

# Hyperparameters
group_dir = './data/groups'
xtcs_list_file = './data/xtcs_list'
strip_parms_list_file = './data/strip_parms_list'
parm_traj_output_path = './data/parm_traj_dict.pkl'

parm_traj_manager = ParmTrajManager(group_dir, xtcs_list_file, strip_parms_list_file)
parm_traj_dict = parm_traj_manager.load_parm_traj_dict(parm_traj_output_path)

dataset_generator = DatasetGenerator(parm_traj_dict, mdcath=True)

#num_residues = n_res_dict(sysname) - 1 

train_loader = dataset_generator.generate_torch_dataloaders(sysname, 'train', batch_size=train_batch_size)
valid_loader = dataset_generator.generate_torch_dataloaders(sysname, 'valid', batch_size=valid_batch_size) 


#num_residues = n_res_dict(sysname) - 1

input_train_instance = list(enumerate(train_loader))[0][1][0]
#print(input_train_instance.size())
ntimesteps = input_train_instance.size(1)
num_residues = input_train_instance.size(2)
#print(input_train_instance.size())

rel_rec, rel_send = rel_send_rec(num_residues)
rel_rec = rel_rec.cuda()
rel_send = rel_send.cuda()
#print(num_residues, rel_rec.shape, rel_send.shape)
encoder = encoder.MLPEncoder(ntimesteps*ndims, n_hidden_encoder, n_edges, do_prob=encoder_dropout, factor=True).to(device)
#encoder = nn.DataParallel(encoder)
decoder = decoder.RNNDecoder(ndims, n_edges, n_hidden_decoder, decoder_dropout, skip_first).to(device)
#decoder = nn.DataParallel(decoder)
#print(encoder, decoder)


optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

best_val_loss = torch.inf
best_epoch = 0


prior = torch.tensor([0.91, 0.09])#, 0.03, 0.03])  # TODO: hard coded for now
print(f"INFO:: Using Prior {prior}")
log_prior = torch.log(prior)
log_prior = torch.unsqueeze(log_prior, 0)
log_prior = torch.unsqueeze(log_prior, 0)
log_prior = log_prior.cuda()


log_file = os.path.join(f'./trained/logs/{sysname}_trainval_log.txt')
log = open(log_file, 'w')

loss_names = ['nll_val', 'nll_train', 'kl_train', 'mse_train', 'acc_train', 'kl_val', 'mse_val', 'acc_val']



to_train = False
trained = False
if os.path.exists(f'./trained/losses/{sysname}_trainval_losses.pkl'):
    losses = pickle.load(open(f'./trained/losses/{sysname}_trainval_losses.pkl', 'rb'))
    if len(losses['nll_val']) < nepochs:
        to_train = True
    else:
        trained = True
        print(f'INFO:: {sysname} trained already! Continuing...')
else:
    to_train = True
    losses = {j:[] for j in loss_names}
# Before starting the training loop

if not to_train and not trained:
    print(f'INFO:: {sysname} not trainable, some issue with the system.')
checkpoint_path = f'./trained/encoder/trained_encoder_{sysname}_*.pt'
latest_epoch = 0
latest_encoder_file = None

# Find the latest encoder checkpoint
for file in glob.glob(checkpoint_path):
    epoch = int(file.split('_')[-1].split('.')[0])  # Extracting epoch number
    if epoch > latest_epoch:
        latest_epoch = epoch
        latest_encoder_file = file
# print(latest_encoder_file)
# Load the latest state dicts if a checkpoint exists
if latest_encoder_file and to_train:
    print(f"INFO:: Loading encoder from {latest_encoder_file}")
    encoder.load_state_dict(torch.load(latest_encoder_file))
    
    # Assuming the decoder follows the same naming pattern with epoch
    latest_decoder_file = latest_encoder_file.replace('encoder', 'decoder')
    if os.path.exists(latest_decoder_file):
        print(f"INFO:: Loading decoder from {latest_decoder_file}")
        decoder.load_state_dict(torch.load(latest_decoder_file))

    # Adjusting optimizer and scheduler if continuing training
    optimizer.load_state_dict(torch.load(latest_encoder_file.replace('encoder', 'optimizer')))
    scheduler.load_state_dict(torch.load(latest_encoder_file.replace('encoder', 'scheduler')))

    # Set the starting epoch for continuing training
    start_epoch = latest_epoch + 1
else:
    start_epoch = 0

# Your training loop
if start_epoch <= 499 and to_train:
    for epoch in range(start_epoch, nepochs):
    
        encoder, decoder, edges_train, probs_train, val_loss, nll_train, kl_train, mse_train, acc_train, kl_val, mse_val, acc_val = train_val.train(encoder, decoder, optimizer, scheduler, train_loader, valid_loader, rel_rec, rel_send, epoch, best_val_loss, ntimesteps, num_residues, log_prior, n_edges, sysname, device, log)

        losses['nll_val'].append(val_loss)
        losses['nll_train'].append(nll_train)
        losses['kl_val'].append(kl_val)
        losses['kl_train'].append(kl_train)
        losses['mse_val'].append(mse_val)
        losses['mse_train'].append(mse_train)
        losses['acc_val'].append(acc_val)
        losses['acc_train'].append(acc_train)

        with open(f'./trained/losses/{sysname}_trainval_losses.pkl', 'wb') as f:
            pickle.dump(losses, f)

        mean_val_loss = torch.mean(torch.tensor(val_loss))
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch

with open(f'./trained/edgesprobs/{sysname}_out_edges_train.pkl', 'wb') as f:
    pickle.dump(edges_train, f)

with open(f'./trained/edgesprobs/{sysname}_out_probs_train.pkl', 'wb') as f:
    pickle.dump(probs_train, f)

with open(f'./trained/losses/{sysname}_trainval_losses.pkl', 'wb') as f:
    pickle.dump(losses, f)

#with open(f'./trained/{sysname}_out_nll_train.pkl', 'wb') as f:
#    pickle.dump(nll_train, f)
#
#with open(f'./trained/{sysname}_out_kl_train.pkl', 'wb') as f:
#    pickle.dump(kl_train, f)
#
#with open(f'./trained/{sysname}_out_mse_train.pkl', 'wb') as f:
#    pickle.dump(mse_train, f)
#
#with open(f'./trained/{sysname}_out_acc_train.pkl', 'wb') as f:
#    pickle.dump(acc_train, f)
#
#with open(f'./trained/{sysname}_out_kl_val.pkl', 'wb') as f:
#    pickle.dump(kl_val, f)
#
#with open(f'./trained/{sysname}_out_mse_val.pkl', 'wb') as f:
#    pickle.dump(mse_val, f)
#
#with open(f'./trained/{sysname}_out_acc_val.pkl', 'wb') as f:
#    pickle.dump(acc_val, f)

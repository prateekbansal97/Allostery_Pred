import numpy as np
import torch
import mdtraj as md
import pickle
import glob
from natsort import natsorted
import os
import subprocess
from tqdm import tqdm


#Hyperparameters

maxlen = 8000
timestep = 64
windows = {'train':[0,int(timestep*0.8)],
            'valid':[int(timestep*0.8), int(timestep*0.9)+1],
            'test':[int(timestep*0.9)+1, int(timestep)-1] 
          }
n_dims = 6


group_list = natsorted(glob.glob('./data/groups/*.txt'))
continued_groups = {f.replace('_continued.txt', '') for f in group_list if '_continued' in f}
final_group_list = [
    f for f in group_list
    if '_continued' in f or f.replace('.txt', '') not in continued_groups
]

with open('./data/xtcs_list', 'r') as g:
    xtcslist = g.readlines()

xtcslist = [j.strip() for j in xtcslist]

with open('./data/strip_parms_list', 'r') as g:
    stripparms = g.readlines()

stripparms = [j.strip() for j in stripparms]
def gen_parm_traj_dict():
    parm_traj_dict = {}
    for group in final_group_list:
        with open(group, 'r') as f:
            lines = f.readlines()
            for line in lines:
                system, projno, run = line.strip().split(',')
                trajs = [j for j in xtcslist if projno in j and f'run{run}' in j]
                if trajs:
                    parm = [j for j in stripparms if system in j]
                    parm_traj_dict[system] = {'trajs':trajs, 'parm':parm[0]}
    pickle.dump(parm_traj_dict, open('./data/parm_traj_dict.pkl','wb'))


def generate_train_valid_test(input_files, parm, sysname):
    for trajno, traj in enumerate(input_files):
        trajname = traj.split('/')[-1].split('.')[0]
        # print(f'Using {traj} as input...')
        pdb = md.load(traj, top=parm)
        CA = pdb.topology.select('name CA')
        pdb = pdb.atom_slice(CA)
        n_residues = len(CA)
        n_frames_per_window = pdb.n_frames // timestep
        if pdb.n_frames % timestep != 0:
            print(sysname, pdb.n_frames)
            closest_multiple_to_timestep_less_than_total_length = (pdb.n_frames // timestep)*timestep
            pdb = pdb[:closest_multiple_to_timestep_less_than_total_length]
        for mode, window in windows.items():
            # print(f'Generating dataset for {mode}ing...')
            start, end = window
            features = np.zeros((end-start, n_frames_per_window, n_residues, n_dims), dtype=np.float64)
            window_start = start
            for nwindow, windowtraj in enumerate(range(start, end)):
                frames_to_choose_for_this_window = np.arange(windowtraj, pdb.n_frames, timestep)
                # if pdb.n_frames < 8000:
                # print(len(frames_to_choose_for_this_window))
                # print(frames_to_choose_for_this_window)
                vel_frames = frames_to_choose_for_this_window + 1
                coords = pdb[frames_to_choose_for_this_window].xyz*10
                # print(coords.shape, n_frames_per_window)
                vels = pdb[vel_frames].xyz*10 - coords
                features[nwindow, :, :, :3] = coords
                features[nwindow, :, :, 3:] = vels
            pickle.dump(features, open(f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{sysname}_{trajname}_{mode}.pkl','wb'))




def gen_dataset_for_all_systems():
    parm_traj_dict = pickle.load(open('./data/parm_traj_dict.pkl','rb'))
    for system_name, trajparm in parm_traj_dict.items():
        trajs, parm = trajparm['trajs'], trajparm['parm']
        trajnames = [traj.split('/')[-1].split('.')[0] for traj in trajs]
        for trajno, trajname in tqdm(enumerate(trajnames), total=len(trajnames), desc=f"Processing {system_name}"):
            for mode, window in windows.items():
                if not os.path.exists(f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{system_name}_{trajname}_{mode}.pkl'):
                    generate_train_valid_test([trajs[trajno]], parm, system_name)
                
#gen_dataset_for_all_systems()



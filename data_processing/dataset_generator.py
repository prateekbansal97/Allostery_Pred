import numpy as np
import mdtraj as md
import pickle
import os
from tqdm import tqdm
from data_processing.utils import load_windows
import glob
from natsort import natsorted

class DatasetGenerator:
    def __init__(self, parm_traj_dict, timestep=64, maxlen=8000, n_dims=6):
        self.parm_traj_dict = parm_traj_dict
        self.timestep = timestep
        self.maxlen = maxlen
        self.n_dims = n_dims
        self.windows = load_windows(timestep)

    def generate_train_valid_test(self, input_files, parm, sysname):
        for trajno, traj in enumerate(input_files):
            trajname = traj.split('/')[-1].split('.')[0]
            pdb = md.load(traj, top=parm)
            CA = pdb.topology.select('name CA')
            pdb = pdb.atom_slice(CA)
            n_residues = len(CA)
            n_frames_per_window = pdb.n_frames // self.timestep

            if pdb.n_frames % self.timestep != 0:
                closest_multiple = (pdb.n_frames // self.timestep) * self.timestep
                pdb = pdb[:closest_multiple]

            for mode, window in self.windows.items():
                start, end = window
                features = np.zeros((end-start, n_frames_per_window, n_residues, self.n_dims), dtype=np.float64)
                for nwindow, windowtraj in enumerate(range(start, end)):
                    frames = np.arange(windowtraj, pdb.n_frames, self.timestep)
                    vel_frames = frames + 1
                    coords = pdb[frames].xyz * 10
                    vels = pdb[vel_frames].xyz * 10 - coords

                    features[nwindow, :, :, :3] = coords
                    features[nwindow, :, :, 3:] = vels

                output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{sysname}_{trajname}_{mode}.pkl'
                pickle.dump(features, open(output_path, 'wb'))

    def generate_combined_train_valid_test(self, sysname, mode):
        output_folder = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data'
        npys = natsorted(glob.glob(f'{output_folder}/*{sysname}*{mode}*'))
        combined = []
        for npy in npys:
            combined.append(pickle.load(open(npy, 'rb')))
        output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{sysname}_{mode}_combined.pkl'
        pickle.dump(combined, open(output_path, 'wb'))



    def process_all_systems(self):
        for system_name, trajparm in self.parm_traj_dict.items():
            trajs, parm = trajparm['trajs'], trajparm['parm']
            trajnames = [traj.split('/')[-1].split('.')[0] for traj in trajs]

            for trajno, trajname in tqdm(enumerate(trajnames), total=len(trajnames), desc=f"Processing {system_name}"):
                for mode, _ in self.windows.items():
                    output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{system_name}_{trajname}_{mode}.pkl'
                    if not os.path.exists(output_path):
                        self.generate_train_valid_test([trajs[trajno]], parm, system_name)
                    output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{system_name}_{mode}_combined.pkl'
                    if not os.path.exists(output_path):
                        self.generate_combined_train_valid_test(system_name, mode)


import numpy as np
import mdtraj as md
import pickle
import os
from tqdm import tqdm
from data_processing.utils import load_windows, check_sizes
import glob
from natsort import natsorted
import torch
import h5py as h5
from os.path import join as opj
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class DatasetGenerator:
    def __init__(self, parm_traj_dict,  mdcath=False, timestep=64, maxlen=8000, n_dims=6):
        self.parm_traj_dict = parm_traj_dict
        self.maxlen = maxlen
        self.n_dims = n_dims
        self.mdcath = mdcath
        self.timestep = timestep
        if self.mdcath:
            self.timestep = 44
        print('timestep: ', self.timestep)
        self.windows = load_windows(self.timestep)



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


    def generate_train_valid_test_mdcath(self, domain_id):
        data_dir = './data/temp/data'
        temperature = '320'
        f = h5.File(opj(data_dir, f"mdcath_dataset_{domain_id}.h5"), 'r')
        pdbProteinAtoms = f[domain_id]['pdbProteinAtoms'][()].decode('utf-8').split('\n')[1:-3] # remove header and footer
        atomtypes = [line.split()[2] for line in pdbProteinAtoms]
        ca_indices = np.where(np.array(atomtypes) == 'CA')[0]
        # print(f'indices of CA atoms: {ca_indices}')
        for replicate in range(5):
            coords = np.array(f[domain_id][temperature][str(replicate)]['coords'])[:,ca_indices,:]
            # print(coords.shape)
            n_residues = len(ca_indices)
            n_frames_per_window = coords.shape[0] // self.timestep

            if coords.shape[0] % self.timestep != 0:
                closest_multiple = (coords.shape[0] // self.timestep) * self.timestep
                coords = coords[:closest_multiple]

            for mode, window in self.windows.items():
                start, end = window
                features = np.zeros((end-start, n_frames_per_window, n_residues, self.n_dims), dtype=np.float64)
                for nwindow, windowtraj in enumerate(range(start, end)):
                    frames = np.arange(windowtraj, coords.shape[0], self.timestep)
                    # print(frames)
                    vel_frames = frames + 1
                    coords2 = coords[frames]
                    vels = coords[vel_frames] - coords2

                    features[nwindow, :, :, :3] = coords2
                    features[nwindow, :, :, 3:] = vels

                output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{domain_id}_rep{replicate}_{mode}.pkl'
                pickle.dump(features, open(output_path, 'wb'))

    def generate_combined_train_valid_test(self, sysname, mode):
        output_folder = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data'
        npys = natsorted(glob.glob(f'{output_folder}/*{sysname}*{mode}*'))
        combined = []
        
        for npy in npys:
            combined.append(pickle.load(open(npy, 'rb')))
        
        output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{sysname}_{mode}_combined.pkl'
        pickle.dump(combined, open(output_path, 'wb'))





    def clean_dataset(self, sysname, mode):
        output_folder = f'./data/processed_data'
        npy = natsorted(glob.glob(f'{output_folder}/*{sysname}*{mode}*combined*'))
        
        dataset = pickle.load(open(npy[0], 'rb'))
        
        all_equal, sizes = check_sizes(dataset)
        
        if not all_equal:
            
            smallest_length_traj = np.min(sizes[:,1])
            dataset = [j[:,:smallest_length_traj,:,:] for j in dataset]
            
            all_equal, _ = check_sizes(dataset)
            
            if all_equal:
                pass
            else:
                print(f"STILL HAS ISSUES {sysname} {mode}")

        features = np.concatenate(dataset, axis=0)

        n_residues = features.shape[2]

        edges = np.zeros((features.shape[0], n_residues, n_residues), dtype=np.int64)
        edges = np.reshape(edges, [-1, n_residues ** 2]) 
        
        off_diag_idx = np.ravel_multi_index(np.where(np.ones((n_residues, n_residues)) - np.eye(n_residues)), [n_residues, n_residues])
        edges = edges[:, off_diag_idx]

        return features, edges

    def normalize_dataset(self, sysname, mode):
        features, edges = self.clean_dataset(sysname, mode)

        min_vals = features.min(axis=3, keepdims=True)  # Shape: [40, 2, 1, 785]
        max_vals = features.max(axis=3, keepdims=True)  # Shape: [40, 2, 1, 785]

        normalized = ((features - min_vals) / (max_vals - min_vals)) * 2 - 1

        
        loc_max = features[:, :, :, 0:3].max()
        loc_min = features[:, :, :, 0:3].min()
        vel_max = features[:, :, :, 3:6].max()
        vel_min = features[:, :, :, 3:6].min()

        loc_train = (features[:, :, :, 0:3] - loc_min) * 2 / (loc_max - loc_min) - 1
        vel_train = (features[:, :, :, 3:6] - vel_min) * 2 / (vel_max - vel_min) - 1

        features_norm = np.concatenate((loc_train, vel_train), axis=3)

        return features_norm, edges



    def generate_torch_dataloaders(self, sysname, mode, batch_size):
        
        features, edges = self.normalize_dataset(sysname, mode)
        
        features = torch.FloatTensor(features)
        edges = torch.LongTensor(edges)

        dataset = TensorDataset(features, edges)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader

    def process_all_systems(self):
        if self.mdcath:
            domainfilename = './data/mdCATH_domains.txt'
            
            with open(domainfilename, 'r') as f:
                domains = [j.strip() for j in f.readlines()]

            import os
            from os.path import join as opj
            from huggingface_hub import HfApi
            from huggingface_hub import hf_hub_download
            from huggingface_hub import hf_hub_url
        
            api = HfApi()


            data_root = './data/temp/'


            pbar = tqdm(domains)
            for domain_id in pbar:
                for mode, _ in self.windows.items():
                    output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{domain_id}_{mode}_combined.pkl'
                    if not os.path.exists(output_path):
                        pbar.set_description(f"Processing {domain_id}")
                        hf_hub_download(repo_id="compsciencelab/mdCATH",
                            filename=f"mdcath_dataset_{domain_id}.h5",
                            subfolder='data',
                            local_dir=data_root,
                            repo_type="dataset")
                # print(f'removing dataset {domain_id}')
                        self.generate_train_valid_test_mdcath(domain_id)
                        os.system(f'rm -rv ./data/temp/data/*{domain_id}*')
                        self.generate_combined_train_valid_test(domain_id, mode)
                        os.system(f'rm -rv ./data/processed_data/{domain_id}_rep*.pkl')
                # for mode, _ in self.windows.items():
                #     output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{domain_id}_{mode}_tensordataset.pkl'
                #     if not os.path.exists(output_path):
                #         self.generate_torch_dataloaders(domain_id, mode)

        
        else:
            for sysname, trajparm in self.parm_traj_dict.items():
                trajs, parm = trajparm['trajs'], trajparm['parm']
                trajnames = [traj.split('/')[-1].split('.')[0] for traj in trajs]

                for trajno, trajname in tqdm(enumerate(trajnames), total=len(trajnames), desc=f"Processing {sysname}"):
                    for mode, _ in self.windows.items():
                        output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{sysname}_{trajname}_{mode}.pkl'
                        if not os.path.exists(output_path):
                            self.generate_train_valid_test([trajs[trajno]], parm, sysname)
                        output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{sysname}_{mode}_combined.pkl'
                        if not os.path.exists(output_path):
                            self.generate_combined_train_valid_test(sysname, mode)
                for mode, _ in self.windows.items():
                    output_path = f'/home/prateek/storage/ML_Allostery/NRI-MD/data/processed_data/{sysname}_{mode}_tensordataset.pkl'
                    if not os.path.exists(output_path):
                        self.generate_torch_dataloaders(sysname, mode)
                #self.torch_tensors_and_dataloaders(sysname, mode)
    def fasta_mdcath(self):
        import requests
        from bs4 import BeautifulSoup
        import pickle
        from tqdm import tqdm
    
        domainfilename = './data/mdCATH_domains.txt'
    
        with open(domainfilename, 'r') as file:
            domains = [j.strip() for j in file.readlines()]
    
        pbar = tqdm(domains)
        with open('./data/mdcath_fasta.fasta', 'w') as fasta_file:
            for domain_id in pbar:
                p = requests.get(f'https://www.cathdb.info/version/latest/domain/{domain_id}/sequence')
                if p.status_code == 200:
                    content = p.text
                    soup = BeautifulSoup(content, 'html.parser')
                    pbar.set_description(f'Processing {domain_id}')
                    sequences = [seq.get_text(strip=True) for seq in soup.find_all("textarea", class_="sequence")]
                    sequences = [s.split('\n')[1:] for s in sequences]
                    sequences = [''.join(s) for s in sequences]
                    seqlens = [len(s) for s in sequences]
    
                    if len(sequences) > 1 and len(set(sequences)) != 1:
                        with open(f'./data/processed_data/{domain_id}_train_combined.pkl', 'rb') as pickle_file:
                            data = pickle.load(pickle_file)
                        resis = data[0].shape[2]
                        seqIndex = [i for i, j in enumerate(seqlens) if j == resis]
                        if len(seqIndex) > 1:
                            print(f"MULTIPLE SEQUENCES WITH THE SAME LENGTHS FOR {domain_id}. CHECK MANUALLY")
                        elif len(seqIndex) == 0:
                            print(f"NO SEQUENCE WITH MATCHING LENGTH FOR {domain_id}. CHECK MANUALLY")
                        else:
                            fasta_file.write(f'''> {domain_id}\n{sequences[seqIndex[0]]}\n''')
                    else:
                        fasta_file.write(f'''> {domain_id}\n{sequences[0]}\n''')
                else:
                    print("Failed to retrieve content.")

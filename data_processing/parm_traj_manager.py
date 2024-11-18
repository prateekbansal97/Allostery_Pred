import pickle
import glob
from natsort import natsorted


class ParmTrajManager:
    def __init__(self, group_dir, xtcs_list_file, strip_parms_list_file):
        self.group_list = natsorted(glob.glob(f'{group_dir}/*.txt'))
        self.continued_groups = {f.replace('_continued.txt', '') for f in self.group_list if '_continued' in f}
        self.final_group_list = [
            f for f in self.group_list
            if '_continued' in f or f.replace('.txt', '') not in self.continued_groups
        ]
        self.xtcs_list = self._load_list(xtcs_list_file)
        self.strip_parms = self._load_list(strip_parms_list_file)

    def _load_list(self, filepath):
        with open(filepath, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def generate_parm_traj_dict(self, output_path):
        parm_traj_dict = {}
        for group in self.final_group_list:
            with open(group, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    system, projno, run = line.strip().split(',')
                    trajs = [j for j in self.xtcs_list if projno in j and f'run{run}' in j]
                    if trajs:
                        parm = [j for j in self.strip_parms if system in j]
                        parm_traj_dict[system] = {'trajs': trajs, 'parm': parm[0]}

        with open(output_path, 'wb') as file:
            pickle.dump(parm_traj_dict, file)

    def load_parm_traj_dict(self, filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)

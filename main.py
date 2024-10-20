import pickle
from data_processing.parm_traj_manager import ParmTrajManager
from data_processing.dataset_generator import DatasetGenerator

# Hyperparameters
group_dir = './data/groups'
xtcs_list_file = './data/xtcs_list'
strip_parms_list_file = './data/strip_parms_list'
parm_traj_output_path = './data/parm_traj_dict.pkl'

# Initialize ParmTrajManager
parm_traj_manager = ParmTrajManager(group_dir, xtcs_list_file, strip_parms_list_file)

# Generate parm_traj_dict and save it
parm_traj_manager.generate_parm_traj_dict(parm_traj_output_path)

# Load parm_traj_dict
parm_traj_dict = parm_traj_manager.load_parm_traj_dict(parm_traj_output_path)

# Initialize DatasetGenerator
dataset_generator = DatasetGenerator(parm_traj_dict)

# Process all systems
dataset_generator.process_all_systems()


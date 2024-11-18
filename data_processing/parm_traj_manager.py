import pickle
import glob
from natsort import natsorted


class ParmTrajManager:
    r"""
    A manager class for handling molecular dynamics parameter and trajectory files.

    This class organizes and processes molecular dynamics (MD) simulation parameter and trajectory files. It 
    identifies and resolves continued groups of files, organizes them into final groups, and creates a dictionary 
    mapping molecular systems to their associated trajectories and parameter files. It also provides methods to save 
    and load this dictionary for reuse.

    Parameters
    ----------
    group_dir : str
        Directory containing text files that group systems into categories.
    xtcs_list_file : str
        Path to a file listing all available trajectory files (.xtc).
    strip_parms_list_file : str
        Path to a file listing all available stripped parameter files (.pdb).

    Attributes
    ----------
    group_list : list of str
        Sorted list of all group files.
    continued_groups : set of str
        Set of group names that have continued versions.
    final_group_list : list of str
        List of final group files, resolving continued versions.
    xtcs_list : list of str
        List of all trajectory file paths loaded from `xtcs_list_file`.
    strip_parms : list of str
        List of all stripped parameter file paths loaded from `strip_parms_list_file`.

    Examples
    --------
    >>> manager = ParmTrajManager('groups', 'xtcs_list.txt', 'strip_parms_list.txt')
    >>> manager.generate_parm_traj_dict('parm_traj_dict.pkl')
    >>> parm_traj_dict = manager.load_parm_traj_dict('parm_traj_dict.pkl')
    """
    
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
        r"""
        Load a list of file paths from a given file.

        Parameters
        ----------
        filepath : str
            Path to the file containing a list of file paths, one per line.

        Returns
        -------
        list of str
            List of file paths.

        Examples
        --------
        >>> manager = ParmTrajManager('groups', 'xtcs_list.txt', 'strip_parms_list.txt')
        >>> xtcs_list = manager._load_list('xtcs_list.txt')
        """
        
        with open(filepath, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def generate_parm_traj_dict(self, output_path):
        r"""
        Generate a dictionary mapping molecular systems to their trajectories and parameter files.

        This method reads the grouped system files, identifies trajectories and parameter files for each 
        system, and saves the resulting dictionary to the specified output path.

        Parameters
        ----------
        output_path : str
            Path to save the generated parameter-trajectory dictionary (in pickle format).

        Saves
        -----
        output_path : dict
            A dictionary with system names as keys and dictionaries containing trajectories ('trajs') and 
            parameter file paths ('parm') as values.

        Examples
        --------
        >>> manager = ParmTrajManager('groups', 'xtcs_list.txt', 'strip_parms_list.txt')
        >>> manager.generate_parm_traj_dict('parm_traj_dict.pkl')
        """
        
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
        r"""
        Load a parameter-trajectory dictionary from a file.

        This method reads a previously saved parameter-trajectory dictionary from a pickle file and returns it.

        Parameters
        ----------
        filepath : str
            Path to the pickle file containing the parameter-trajectory dictionary.

        Returns
        -------
        dict
            The loaded parameter-trajectory dictionary.

        Examples
        --------
        >>> manager = ParmTrajManager('groups', 'xtcs_list.txt', 'strip_parms_list.txt')
        >>> parm_traj_dict = manager.load_parm_traj_dict('parm_traj_dict.pkl')
        """

        
        
        with open(filepath, 'rb') as file:
            return pickle.load(file)

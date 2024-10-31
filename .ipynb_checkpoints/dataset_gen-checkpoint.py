import time
import numpy as np
import argparse
from copy import deepcopy
from scipy import interpolate

parser = argparse.ArgumentParser('Preprocessing: Generate training/validation/testing features from pdb')
parser.add_argument('--MDfolder', type=str, default="data/pdb/",
                    help='folder of pdb MD')
parser.add_argument('--pdb-start', type=int, default="1",
                    help='select pdb file window from start, e.g. in tutorial it is ca_1.pdb')
parser.add_argument('--pdb-end', type=int, default="56",
                    help='select pdb file window to end')
parser.add_argument('--num-residues', type=int, default=77,
                    help='Number of residues of the MD pdb')
parser.add_argument('--feature-size', type=int, default=6,
                    help='The number of features used in study( position (X,Y,Z) + velocity (X,Y,Z) ).')
parser.add_argument('--train-interval', type=int, default=60,
                    help='intervals in trajectory in training')
parser.add_argument('--validate-interval', type=int, default=60,
                    help='intervals in trajectory in validate')
parser.add_argument('--test-interval', type=int, default=100,
                    help='intervals in trajectory in test')
args = parser.parse_args()


def convert_dataset_md_single(MDfolder, startIndex, experiment_size, timestep_size, feature_size, num_residues, interval, pdb_start, pdb_end, aa_start, aa_end):
    """
    Convert in single md file in single skeleton
    """
    features = list()                       
    edges = list()
    
    for i in range(startIndex, experiment_size+1):
        print("Start: "+str(i)+"th PDB")
        for j in range(pdb_start, pdb_end+1): #PDB start startes the sliding window at the first frame, the end frame (the last start window), is 56 (pdb_end)
            # print(str(i)+" "+str(j))
            features.append(read_feature_MD_file_slidingwindow(MDfolder+"ca_"+str(
                i)+".pdb", timestep_size, feature_size, num_residues, interval, j, aa_start, aa_end))
            edges.append(np.zeros((num_residues, num_residues)))
    print("***")
    print(len(features))
    print("###")
    features = np.stack(features, axis=0)
    edges = np.stack(edges, axis=0)

    return features, edges
def read_feature_MD_file_slidingwindow(filename, timestep_size, feature_size, num_residues, interval, window_choose, aa_start, aa_end):
    # read single expriments of all time points
    feature = np.zeros((timestep_size, feature_size, num_residues)) # 50 timesteps for each sliding window trajectory (3000 frames in the trajectory)

    #feature size is 6 to account for 3 positions and 3 velocities, train_interval is the gap between two timesteps between the sliding window trajectory (60 in this case).

    flag = False
    nflag = False
    modelNum = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split()
            if(line.startswith("MODEL")): 
                modelNum = int(words[1])
                if (modelNum % interval == window_choose):
                    # print(f'MODELNUM for FLAG TRUE: {modelNum}, window_choose = {window_choose}, interval = {interval}')
                    flag = True
                if (modelNum % interval == (window_choose+1)):
                    # print(f'MODELNUM for nFLAG TRUE: {modelNum}, window_choose = {window_choose}, interval = {interval}')
                    nflag = True
            elif(line.startswith("ATOM") and words[2] == "CA" and int(words[4]) >= aa_start and int(words[4]) <= aa_end and flag):
                numStep = int(modelNum/interval)
                feature[numStep, 0, int(words[4])-aa_start] = float(words[5])
                feature[numStep, 1, int(words[4])-aa_start] = float(words[6])
                feature[numStep, 2, int(words[4])-aa_start] = float(words[7])
            elif(line.startswith("ATOM") and words[2] == "CA" and int(words[4]) >= aa_start and int(words[4]) <= aa_end and nflag):
                numStep = int(modelNum/interval)
                feature[numStep, 3, int(
                    words[4])-aa_start] = float(words[5])-feature[numStep, 0, int(words[4])-aa_start]
                feature[numStep, 4, int(
                    words[4])-aa_start] = float(words[6])-feature[numStep, 1, int(words[4])-aa_start]
                feature[numStep, 5, int(
                    words[4])-aa_start] = float(words[7])-feature[numStep, 2, int(words[4])-aa_start]
            elif(line.startswith("ENDMDL") and flag):
                flag = False
            elif(line.startswith("ENDMDL") and nflag):
                nflag = False
    f.close()
    # print(feature.shape)
    return feature

MDfolder = args.MDfolder
feature_size = args.feature_size
num_residues = args.num_residues
pdb_start = args.pdb_start
pdb_end = args.pdb_end
train_interval = args.train_interval
validate_interval = args.validate_interval
test_interval = args.test_interval

print("Generate Train")
features, edges = convert_dataset_md_single(MDfolder, startIndex=1, experiment_size=1, timestep_size=50,
                                            feature_size=feature_size, num_residues=num_residues, interval=train_interval, pdb_start=pdb_start, pdb_end=42, aa_start=1, aa_end=num_residues)

np.save('data/features.npy', features)
np.save('data/edges.npy', edges)


print("Generate Valid")
features_valid, edges_valid = convert_dataset_md_single(MDfolder, startIndex=1, experiment_size=1, timestep_size=50,
                                                        feature_size=feature_size, num_residues=num_residues, interval=validate_interval, pdb_start=43, pdb_end=52, aa_start=1, aa_end=num_residues)

np.save('data/features_valid.npy', features_valid)
np.save('data/edges_valid.npy', edges_valid)


print("Generate Test")
features_test, edges_test = convert_dataset_md_single(MDfolder, startIndex=1, experiment_size=1, timestep_size=50,
                                                      feature_size=feature_size, num_residues=num_residues, interval=test_interval, pdb_start=53, pdb_end=59, aa_start=1, aa_end=num_residues)
np.save('data/features_test.npy', features_test)
np.save('data/edges_test.npy', edges_test)


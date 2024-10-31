import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import mplhelv
from data_processing.utils import load_windows, n_res_dict, rel_send_rec
from tqdm import tqdm

#plt.rc('lines', linewidth=2.5)
#plt.rc('axes', prop_cycle=plt.cycler('color', ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']))
plt.rc('font', size=12)
plt.rc('axes', linewidth=1.5)
plt.rc('xtick.major', width=1.5)
plt.rc('ytick.major', width=1.5)
plt.rc('axes', titlesize=12)
#plt.rc('axes.spines', top=False, right=False)
plt.rc('legend', frameon=False)


sysname = '5ht4r_7XT8'

edges = f'{sysname}_out_edges_train.pkl'
probs = f'{sysname}_out_probs_train.pkl'


num_residues = n_res_dict(sysname) - 1
windowsize = 255 


def pickle_load(filename):
    with open(f'./data/trained/{filename}', 'rb') as f:
        output = pickle.load(f)
    return output


def getEdgeResults(probs, num_residues, windowsize, threshold=False):
    probs = probs[:, :, 1]
    
    residueR2 = num_residues*(num_residues-1)
    probs = np.reshape(probs, (windowsize, residueR2))
    
    edges_train = probs/windowsize

    results = np.zeros((residueR2))
    for i in range(windowsize):
        results = results+edges_train[i, :]

    if threshold:
        # threshold, default 0.6
        index = results < (threshold)
        results[index] = 0

    # Calculate prob for figures
    edges_results = np.zeros((num_residues, num_residues))
    count = 0
    for i in range(num_residues):
        for j in range(num_residues):
            if not i == j:
                edges_results[i, j] = results[count]
                count += 1
            else:
                edges_results[i, j] = 0
    edges_results = edges_results - edges_results.min() / (edges_results.max() - edges_results.min())
    return edges_results

def is_symmetric(array):
  return np.allclose(array, array.T)

edges = pickle_load(edges)
probs = pickle_load(probs)
for threshold in tqdm(np.arange(0, 0.15, 0.01)):
    fig, axs = plt.subplots()
    results = getEdgeResults(probs, num_residues, windowsize, threshold=threshold)
    print('SYMMETRIC?', is_symmetric(results), threshold)
    mat = axs.imshow(results, cmap='binary', origin='lower')
    cbar = fig.colorbar(mat)
    axs.set_xlabel('Receivers (residue numbers)')
    axs.set_ylabel('Senders (residue numbers)')
    #axs.annotate('', xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=1.5))
    #axs.annotate('', xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=1.5))

    fig.savefig(f'./data/trained/{sysname}_trainedgraph_{int(threshold*100)}.png', dpi=300)
    plt.close('all')
print(edges.shape, probs.shape, results.shape)


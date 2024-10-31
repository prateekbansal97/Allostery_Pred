import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplhelv
from natsort import natsorted
import glob
import re
plt.rc('lines', linewidth=2.5)
plt.rc('axes', prop_cycle=plt.cycler('color', ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']))
plt.rc('font', size=12)
plt.rc('axes', linewidth=1.5)
plt.rc('xtick.major', width=1.5)
plt.rc('ytick.major', width=1.5)
plt.rc('axes', titlesize=12)
plt.rc('axes.spines', top=False, right=False)
plt.rc('legend', frameon=False)

# Load the .pkl files
def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def plot_system_acc_train(sysname, data=None):
    
    filepath = f'./trained/losses/{sysname}_trainval_losses.pkl'
    
    if not data:
        data = load_pkl(filepath)
        print(list(data.keys()))
        data = {i: np.mean(np.array(j), axis=1) for i, j in data.items()}
        print([(i, j.shape) for i, j in data.items()])
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot labels and titles
    plot_info = [
        ('acc_train', 'acc_val', 'Edge Accuracy', 'Edge Accuracy (%)'),
        ('kl_train', 'kl_val', 'Kullback - Liebler Divergence', 'Average KL-Divergence Error'),
        ('mse_train', 'mse_val', 'Mean Squared Error', 'Average MSE'),
        ('nll_train', 'nll_val', 'Negative Log Likelihood', 'Average NLL')
    ]
    
    # Plotting loop
    for i, (train_key, val_key, title, ylabel) in enumerate(plot_info):
        ax = axs.flat[i]
        if train_key == "kl_train":
            ax.axhline(0, 0, len(data[train_key]), linewidth=1.5, ls='--', color='k')
            data[val_key] *= -1
        ax.plot(data[train_key] * 100 if 'acc' in train_key else data[train_key], label='Train')
        ax.plot(data[val_key] * 100 if 'acc' in val_key else data[val_key], label='Validation')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Epoch')
        ax.set_xlim(-1, len(data[train_key]))
        ax.legend()
        ax.text(-0.1, 1.1, chr(97 + i), transform=ax.transAxes, size=12, weight='bold')
        if train_key == "kl_train":
            ax.axhline(0, 0, len(data[train_key]), linewidth=1.5, ls='--', color='k')
    # Set overall plot properties
    fig.tight_layout()
    fig.savefig(f'./trained/images/{sysname}_training_validation_loss.png', dpi=300)


def plot_if_no_losspkl(sysname):
    logfile = f'./trained/logs/{sysname}_trainval_full_log.txt'
    try: 
        with open(logfile, 'r') as f:
            log_data = f.readlines()
    except FileNotFoundError:
        logfile = f'./trained/logs/{sysname}_trainval_log.txt'
        with open(logfile, 'r') as f:
            log_data = f.readlines()

    log_data = [j.strip() for j in log_data]
    
    # Initialize lists for metrics
    epochs, nll_train, kl_train, mse_train, acc_train = [], [], [], [], []
    nll_val, kl_val, mse_val, acc_val = [], [], [], []
    
    # Regex pattern to extract data
    pattern = r"Epoch: (\d+) nll_train: ([\d\.]+) kl_train: ([\d\.]+) mse_train: ([\d\.]+) acc_train: ([\d\.]+) nll_val: ([\d\.]+) kl_val: ([\-\d\.]+) mse_val: ([\d\.]+) acc_val: ([\d\.]+)"
    
    keys = ['nll_val', 'nll_train', 'kl_train', 'mse_train', 'acc_train', 'kl_val', 'mse_val', 'acc_val']
    data = {i:[] for i in keys}

    # Extract data from log lines
    for line in log_data:
        match = re.search(pattern, line)
        if match:
            data['nll_train'].append(float(match.group(2)))
            data['kl_train'].append(float(match.group(3)))
            data['mse_train'].append(float(match.group(4)))
            data['acc_train'].append(float(match.group(5)))
            data['nll_val'].append(float(match.group(6)))
            data['kl_val'].append(-1*float(match.group(7)))
            data['mse_val'].append(float(match.group(8)))
            data['acc_val'].append(float(match.group(9)))
    plot_system_acc_train(sysname, data)

sysnames_processed = []
done_systems = natsorted(glob.glob('./trained/losses/*trainval_losses*'))
for loss in done_systems:
    sysname = '_'.join(loss.split('/')[-1].split('_')[:2])
    plot_system_acc_train(sysname)
    sysnames_processed.append(sysname)

done_systems2 = natsorted(glob.glob('./trained/encoder/*.pt'))
done_systems2 = natsorted(list(set(['_'.join(j.split('/')[-1].split('_')[2:4]) for j in done_systems2])))
    
for sysname in done_systems2:
    if sysname not in sysnames_processed:
        plot_if_no_losspkl(sysname)

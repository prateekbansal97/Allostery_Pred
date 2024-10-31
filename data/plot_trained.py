import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplhelv

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

sysname = "5ht1e_7E33"
filepath = f'./trained/{sysname}_trainval_losses.pkl'

data = load_pkl(filepath)
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
fig.savefig(f'./{sysname}_training_validation_loss.png', dpi=300)


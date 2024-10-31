import re
import matplotlib.pyplot as plt
import mplhelv
import matplotlib.ticker as mticker

# Set the x and y axis to only have whole number ticks
plt.rc('lines', linewidth=2.5)
plt.rc('axes', prop_cycle=plt.cycler('color', ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']))
plt.rc('font', size=12)
plt.rc('axes', linewidth=1.5)
plt.rc('xtick.major', width=1.5)  
plt.rc('ytick.major', width=1.5)
plt.rc('axes', titlesize=12)
plt.rc('axes.spines', top=False, right=False)
plt.rc('legend', frameon=False)
# Log data as a list of strings (or load it from a file)
log_data = [
    "Epoch: 0000 nll_train: 1371303624.0000000000 kl_train: 490.5704803467 mse_train: 184.3150053024 acc_train: 0.4828207789 nll_val: 4166472.8055555555 kl_val: -535.0803561740 mse_val: 0.5600097775 acc_val: 1.0000000000 time: 48.2029s",
    "Epoch: 0001 nll_train: 1326480856.0000000000 kl_train: 484.7265491486 mse_train: 178.2904386520 acc_train: 0.5294081686 nll_val: 4031887.6111111110 kl_val: -535.0375569661 mse_val: 0.5419203904 acc_val: 0.6735919891 time: 48.3876s",
    # Add more lines here
]

sysname = '5ht4r_7XT8'
logfile = f'./trained/{sysname}_trainval_full_log.txt'

with open(logfile, 'r') as f:
    log_data = f.readlines()

log_data = [j.strip() for j in log_data]

# Initialize lists for metrics
epochs, nll_train, kl_train, mse_train, acc_train = [], [], [], [], []
nll_val, kl_val, mse_val, acc_val = [], [], [], []

# Regex pattern to extract data
pattern = r"Epoch: (\d+) nll_train: ([\d\.]+) kl_train: ([\d\.]+) mse_train: ([\d\.]+) acc_train: ([\d\.]+) nll_val: ([\d\.]+) kl_val: ([\-\d\.]+) mse_val: ([\d\.]+) acc_val: ([\d\.]+)"

# Extract data from log lines
for line in log_data:
    match = re.search(pattern, line)
    if match:
        epochs.append(int(match.group(1)))
        nll_train.append(float(match.group(2)))
        kl_train.append(float(match.group(3)))
        mse_train.append(float(match.group(4)))
        acc_train.append(float(match.group(5)))
        nll_val.append(float(match.group(6)))
        kl_val.append(float(match.group(7)))
        mse_val.append(float(match.group(8)))
        acc_val.append(float(match.group(9)))

# Plot the metrics
plt.figure(figsize=(12, 8))

# Subplot 1: NLL Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, nll_train, label='NLL Train')
plt.plot(epochs, nll_val, label='NLL Val')
# Set x and y limits to whole numbers
plt.xlim(0, 200)#int(plt.xlim()[0]), int(plt.xlim()[1]) + 1)  # Add 1 to ensure whole number
plt.ylim(0, 5*1e5)#int(plt.ylim()[0]), int(plt.ylim()[1]) + 1)
plt.title('Negative Log Likelihood Loss')
plt.xlabel('Epoch')
plt.ylabel('NLL')
plt.legend()

# Subplot 2: KL Divergence
plt.subplot(2, 2, 2)
plt.plot(epochs, kl_train, label='KL Train')
plt.plot(epochs, kl_val, label='KL Val')
plt.title('Kullback–Leibler Divergence')
# Set x and y limits to whole numbers
plt.axhline(0, 0, 200, linewidth=1.5, ls='--', color='k')
plt.xlim(0, 200)
plt.ylim(-300, 300)#int(plt.ylim()[0]), int(plt.ylim()[1]) + 1)

plt.xlabel('Epoch')
plt.ylabel('K-L Divergence')
plt.legend()

# Subplot 3: MSE Loss
plt.subplot(2, 2, 3)
plt.plot(epochs, mse_train, label='MSE Train')
plt.plot(epochs, mse_val, label='MSE Val')
# Set x and y limits to whole numbers
plt.xlim(0, 200)#int(plt.xlim()[0]), int(plt.xlim()[1]) + 1)  # Add 1 to ensure whole number
plt.ylim(0.0, 0.06)#int(plt.ylim()[0]), int(plt.ylim()[1]) + 1)
plt.title('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()

# Subplot 4: Accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, acc_val, label='Acc Val', color='#6B5B95')
plt.plot(epochs, acc_train, label='Acc Train', color='#FF6F61')
# Set x and y limits to whole numbers
plt.xlim(0, 200)#int(plt.xlim()[0]), int(plt.xlim()[1]) + 1)  # Add 1 to ensure whole number
plt.ylim(0, 1.02)#int(plt.ylim()[0]), int(plt.ylim()[1]) + 1)

plt.title('Edge Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(f'./{sysname}_train_valid_plot.png',dpi=300)
#plt.tight_layout()
#plt.show()



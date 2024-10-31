import os
import subprocess
import pickle


done_systems = './trained/done_systems'
with open(done_systems, 'r') as f:
    done_systems = [j.strip().split(',')[0] for j in f.readlines()]


submit_bash_script = './submit_bash_script.bash'

os.chdir('./data/processed_data')
proc = subprocess.Popen('ls -1S', shell=True, stdout=subprocess.PIPE)
output = proc.communicate()[0]
sysnames = ['_'.join(j.split('_')[:2]) for j in output.decode('utf-8').split('\n') if 'train_combined' in j]
trimmed = []
for j in sysnames:
    if 'train' in j:
        trimmed.append(j.split('_')[0])
    else:
        trimmed.append(j)
sysnames = trimmed.copy()
sysnames.reverse()
#print(sysnames)
os.chdir('../../')
with open(submit_bash_script, 'w') as f:
    f.write(f"""source /home/prateek/miniconda3/bin/activate pt\n""")
    for sysname in sysnames:
        if sysname not in done_systems:
            if os.path.exists(f'./data/processed_data/{sysname}_test_combined.pkl'):
                if not os.path.exists(f'./trained/losses/{sysname}_trainval_losses.pkl'):
                    f.write(f"""python main.py --sysname {sysname} &> ./trained/logs/{sysname}_trainval_full_log.txt && echo 'Job finished for {sysname}' | mail -s '{sysname} NRI MODEL TRAINED Notification' pdb3@illinois.edu\n""")
                else:
                    losses = pickle.load(open(f'./trained/losses/{sysname}_trainval_losses.pkl', 'rb'))
                    if len(losses['nll_val']) < 500:
                        f.write(f"""python main.py --sysname {sysname} &> ./trained/logs/{sysname}_trainval_full_log.txt && echo 'Job finished for {sysname}' | mail -s '{sysname} NRI MODEL TRAINED Notification' pdb3@illinois.edu\n""")
                    else:
                        continue
                #f.write(f'

import os


with open('../sorted_DONE_SYSTEMS','r') as f:
    done = f.readlines()
done = [l.rstrip().lstrip() for l in done]
total_sys = []
total_groups = []
for i in range(40,360,10):
    cont, nocont = 0,0
    if os.path.exists(f'group_{i}_{i+10}_continued'):
        with open(f'group_{i}_{i+10}_continued','r') as f:
            lines=f.readlines()
            cont = 1
    elif os.path.exists(f'group_{i}_{i+10}'):
        with open(f'group_{i}_{i+10}','r') as f:
            lines=f.readlines()
            nocont = 1
    else:
        continue
    nsys = len(lines)
    count=0
    for line in lines:
        #print(line.split(','))
        system = line.split(',')[0]
        if system in done:
            count+=1
    if nsys == count:
        for line in lines:
            total_sys.append([line.split(',')[0],line.split(',')[1]])
        print(line.split(',')[1], 'is done', f'group_{i}_{i+10}')
        if cont==1:
            total_groups.append(os.path.abspath(f'group_{i}_{i+10}_continued'))
        elif nocont==1:
            total_groups.append(os.path.abspath(f'group_{i}_{i+10}'))
print(total_groups)
print(total_sys)

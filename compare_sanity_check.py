import numpy as np
import glob
import pickle
from tqdm import tqdm


path2 = './data/processed_data/'
path1 = './data/processed_data2/'

npys1 = glob.glob(f'{path1}/*.pkl')
npys2 = glob.glob(f'{path2}/*.pkl')

for npy in tqdm(npys1):
    npyname = npy.split('/')[-1]
    for npy2 in npys2:
        npyname2 = npy2.split('/')[-1]
        if npyname == npyname2:
            data1 = pickle.load(open(npy, 'rb'))
            data2 = pickle.load(open(npy2, 'rb'))
            comparison = data1 == data2
            equal_arrays = comparison.all()
            if not equal_arrays:
                print(npy, npy2)

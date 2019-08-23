import argparse
import numpy as np
import pdb
from models.utils import many_one_hot
import h5py
from tqdm import tqdm
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import MolFromSmiles

parser = argparse.ArgumentParser()
parser.add_argument('--molecular_weight', type=int, default=200)
args = parser.parse_args()


f = open('data/250k_rndm_zinc_drugs_clean.smi','r')

L = []
chars = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[', '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/', '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
DIM = len(chars)
for line in tqdm(f):
    line = line.strip()
    if ExactMolWt(MolFromSmiles(line))< args.molecular_weight:
        L.append(line)
f.close()

count = 0
MAX_LEN = 120
OH = np.zeros((249456,MAX_LEN,DIM))
for chem in tqdm(L):
    indices = []
    for c in chem:
        indices.append(chars.index(c))
    if len(indices) < MAX_LEN:
        indices.extend((MAX_LEN-len(indices))*[DIM-1])
    OH[count,:,:] = many_one_hot(np.array(indices), DIM)
    count = count + 1
f.close()
h5f = h5py.File('zinc_str_dataset.h5','w')
h5f.create_dataset('data', data=OH)
h5f.create_dataset('chr',  data=chars)
h5f.close()

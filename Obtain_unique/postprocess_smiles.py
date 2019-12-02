# ------------------------------------------------------
import numpy as np
import scipy as sc
import sys, string, os 
import glob 
# ------------------------------------------------------
def obtain_unique_smiles(fname, smiles_set):
# ------------------------------------------------------
    with open(fname, 'r') as fp:
        smiles_data = fp.read().splitlines()
# ------------------------------------------------------
    smiles = []
    for i in range(0, len(smiles_data)):
        smiles.append(str(smiles_data[i]).strip(' ')) 
# ------------------------------------------------------
    #smiles = list(set(smiles)) 

    smiles_set_diff = list(set(smiles) - set(smiles_set))     

    #print(set(smiles) - set(smiles_set))

    for i in range(0, len(smiles_set_diff)):
        smiles_set.append(smiles_set_diff[i])    
# ------------------------------------------------------
    return smiles_set 
# -------------------------------------------------------
if __name__ == '__main__':

    smiles = []

    all_files = glob.glob('*.smi')


    for f in range(0, len(all_files)):
        fname = all_files[f] 
        print('File No. = ', f)
        smiles = obtain_unique_smiles(fname, smiles)


    with open('Generated_unique_smiles.smi', 'w') as fp:
        for j in range(0, len(smiles)):
            st = smiles[j] + "\n"
            fp.writelines("{}".format(st)) 



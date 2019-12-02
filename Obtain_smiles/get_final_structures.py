# -------------------------------------------------------------------
import numpy as np
import scipy as sc
import sys, string, os, glob
import pickle  
# -------------------------------------------------------------------
def get_filesize(): 
# -------------------------------------------------------------------
    fname = 'Generated_unique_smiles.smi' 
    with open(fname, 'r') as fp:
        smiles_data = fp.read().splitlines()   

    files = glob.glob('*.svg') 

    considered_files = [] 
    considered_smiles = [] 
    
    for d in range(0, len(files)):
        f = files[d]
        get_size = os.path.getsize(f)
        
        if get_size > 0:
            indx = np.int_(f[:-4])
            considered_files.append(f) 
            smiles = str(smiles_data[indx]).strip(' ')
            considered_smiles.append(smiles)
        else:
            cmd = 'rm ' + f 
            os.system(cmd)
     
    print(len(considered_files)) 


    with open('Considered_smiles.smi', 'w') as fp:
        for i in range(0, len(considered_files)):
             fp.write("{}\n".format(considered_smiles[i])) 



if __name__ == "__main__":
    
    get_filesize() 


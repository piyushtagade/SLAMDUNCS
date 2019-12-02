# -------------------------------------------------------------------
import numpy as np
import scipy as sc
import sys, string, os
# -------------------------------------------------------------------
def validate_structures(fname): 
# -------------------------------------------------------------------
    with open(fname, 'r') as fp:
        smiles_data = fp.read().splitlines() 
# -------------------------------------------------------------------
    command = "obabel -:"
    for i in range(0, len(smiles_data)):
        smiles = str(smiles_data[i]).strip(' ')
         
        s = '"' + smiles + '"'       
        op = str(i)+'.svg'
        cmd = command + s + ' -O ' + op 

        try:  
            os.system(cmd)             
        except:
            print('Null')
        #print(cmd) 

if __name__ == "__main__":
    
    validate_rbm('Generated_unique_smiles.smi') 


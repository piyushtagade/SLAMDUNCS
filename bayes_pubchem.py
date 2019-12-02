# -----------------------------------------------------------------------------
# Deep Learning Bayesian framework for inverse materials design
# -----------------------------------------------------------------------------
import scipy as sc
import numpy as np
# -----------------------------------------------------------------------------
from pprint import pprint 
# Load the dataset
#from deep_learning import *
from deep_neural_network import * 
from deep_bayesian_inference import *
from process_smiles import * 
from pubchem_data import * 
import pickle
from anova import *
import time
from train_rbm import *
from train_dnn import *

# -----------------------------------------------------------------------------
# For ensuring that same random number is always generated 
# -----------------------------------------------------------------------------
np.random.seed(105)

mxl = 100 
# -----------------------------------------------------------------------------
# Training the rbm for unsupervised learning of molecular structures
# -----------------------------------------------------------------------------
pubchem_data_filename = 'Molecular_data_pubchem'
train_rbm(pubchem_data_filename, max_smiles_length=100)
with open('rbm_trained.pkl', 'rb') as fp:
    rbm = pickle.load(fp)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Training the dnn for semi-supervised learning of structure-property 
# correlation
# -----------------------------------------------------------------------------
properties_data_filename = 'IQR_screened_redox_homolumo'
train_dnn(pubchem_data_filename, properties_data_filename, 
              max_smiles_length=100)

with open('dnn_trained.pkl', 'rb') as fp:
    dnn = pickle.load(fp)

# -----------------------------------------------------------------------------
# Creating a Deep Bayesian Inference object
# -----------------------------------------------------------------------------

dbi = Deep_Bayesian_Inference(dnn, rbm)

max_smiles_length = 100

pah_smiles = [] 
fixed_smiles = [] 
copy_loc_indes = [] 

# -----------------------------------------------------------------------------
# Some sample smiles for initialization of Markov Chain
# -----------------------------------------------------------------------------
pah_smiles.append('C1=CC2=C3C(=CC=C4C3=C1C5=C6C4=CC=C7C6=C(C=C5)C(=O)NC7=O)C(=O)NC2=O')
pah_smiles.append('O=C1NC(=O)C2=C3C1=CC=C1C(=O)NC(=O)C(C=C2)=C31')

# -----------------------------------------------------------------------------
# Number of chains to run
# -----------------------------------------------------------------------------
num_parallel_chains = 5000 
# -----------------------------------------------------------------------------
# Randomly creating initial state of the Markov Chain
# -----------------------------------------------------------------------------
ini_smiles = []
fixed_smiles_index = []
copy_loc_index = []  
for par in range(0, num_parallel_chains):
    if np.random.random(1) > 0.00:
       i_pah = np.random.choice(len(pah_smiles), 1)[0]
       
       #i_pah = 0
       if i_pah < 0:
           if np.random.random(1) > 0.05:
               side_add = 'both'
           else:
               side_add = 'one'               
           if np.random.random(1) > 0.05:
               symmetry = True
           else:
               symmetry = False   
           print('Before initialize') 
            
           updated_smiles, fixed_index, copy_loc = initialize_smiles(pah_smiles[i_pah], symmetry=symmetry, side_add = side_add)
           print(updated_smiles)
       else:

           if i_pah == 1:  
               i_rand = np.random.random(1)
           
               if i_rand < 0.3: 
                   updated_smiles, fixed_index, copy_loc =  initialize_smiles_dimer(pah_smiles[1], symmetry=True, add_side = 1)
               if i_rand >= 0.3 and i_rand < 0.6:
                   updated_smiles, fixed_index, copy_loc = initialize_smiles_dimer(pah_smiles[1], symmetry=True, add_side = 2)
               if i_rand >= 0.6 and i_rand < 0.9:
                   updated_smiles, fixed_index, copy_loc = initialize_smiles_dimer(pah_smiles[1], symmetry=True, add_side = 0)
               if i_rand > 0.9:
                   updated_smiles, fixed_index, copy_loc = initialize_smiles_dimer(pah_smiles[1], symmetry=False, add_side = 2)
          
           else:
               i_rand = np.random.random(1)
               if i_rand < 0.3:
                   updated_smiles, fixed_index, copy_loc =  initialize_smiles_dimer(pah_smiles[0], symmetry=True, add_side = 1, rings=2)
               if i_rand >= 0.3 and i_rand < 0.6:
                   updated_smiles, fixed_index, copy_loc = initialize_smiles_dimer(pah_smiles[0], symmetry=True, add_side = 2, rings=2)
               if i_rand >= 0.6 and i_rand < 0.9:
                   updated_smiles, fixed_index, copy_loc = initialize_smiles_dimer(pah_smiles[0], symmetry=True, add_side = 0, rings=2)
               if i_rand > 0.9:
                   updated_smiles, fixed_index, copy_loc = initialize_smiles_dimer(pah_smiles[0], symmetry=False, add_side = 2, rings=2)
 
    else: 
        if np.random.rand(1) > 0.0: 
           i_pah = np.random.choice(len(indx_lumo), 1)[0]
           print(i_pah)
           updated_smiles, fixed_index, copy_loc = initialize_smiles(smiles1[indx_lumo[i_pah]], symmetry=False, side_add = 'both')
        else:
           i_pah = np.random.choice(len(smiles_intrinsic), 1)[0]
           #updated_smiles, fixed_index, copy_loc = initialize_smiles(smiles_intrinsic[i_pah], symmetry=False, side_add = 'none')
           updated_smiles = smiles_intrinsic[i_pah]
           change_index = []
           try:
               cstart = np.random.choice(len(updated_smiles)-20, 1, replace=False)        
           except:
               cstart = 0 
        
           cend = cstart + np.random.choice(18, 1, replace=False) + 1 

           change_index.append(np.arange(cstart, cend, 1))

           fixed_index = np.setdiff1d(np.array([i for i in range(0, len(updated_smiles))]), np.array(change_index))
           copy_loc = []  
        
    ini_smiles.append(updated_smiles)
    fixed_smiles_index.append(fixed_index)
    copy_loc_index.append(copy_loc)
               
# -----------------------------------------------------------------------------    

list_of_mxl = [] 

ini_samp = np.zeros((1,  max_smiles_length*8), dtype = np.int)

par_samp = []

indx_db = np.array([2, 5, 9, 13, 17])
find_db = []
#indx = np.random.choice(int(len(training_data)), int(num_parallel_chains), replace = False)
for i in range(0, num_parallel_chains): 
    #loc_db = [pos for pos, char in enumerate(pah_smiles[i]) if char == '=']
    #find_db.append(loc_db)
    list_of_mxl.append(min(len(ini_smiles[i]), max_smiles_length))
    st = ini_smiles[i] #+ 'CCCCC'    
    b = smiles_to_binary(st, max_smiles_length)
    par_samp.append(np.reshape(b, [1, -1]))     
# -----------------------------------------------------------------------------
# Options for Bayesian inference     
# -----------------------------------------------------------------------------
chain_length = 50000
burnout_period = 10000     
dbi.options(burnout_period, chain_length)

data = np.zeros(2); sdev = np.zeros(2)
# -----------------------------------------------------------------------------
# Running the Markov Chain Monte Carlo sampling algorithm
# -----------------------------------------------------------------------------
homo_state_all = []; lumo_state_all = [] 
#fname = 
#cons_chain = [76, 83, 97]
for par_chain in range(0, num_parallel_chains):
    print('--------------------------------------------------------------------')
    print('                      chain number = ', par_chain)
    print('--------------------------------------------------------------------')
   
    try: 
        change_smiles_index = [] 

        smiles_index = np.array([i for i in range(0, list_of_mxl[par_chain])]) 

        change_smiles_index.append(np.setdiff1d(smiles_index, fixed_smiles_index[par_chain])) 

        print(change_smiles_index)
    
    
        start_time = time.time()
        fname = 'Generated_smiles_bay_lumo_with_fixed_pah' + str(par_chain) + '_new.smi'
    #ini_samp = np.zeros((1,  list_of_mxl[par_chain]*8), dtype = np.int)
        ini_samp[0,:] = par_samp[par_chain][:]
        post_samples, accepted_samples, homo_state, lumo_state, visited_pred = dbi.inference(data, sdev, ini_samp, change_index = change_smiles_index, copy_loc=copy_loc_index[par_chain])
        homo_state_all.append(homo_state); lumo_state_all.append(lumo_state)
#post_samples, accepted_samples = dbi.inference(data, sdev, ini_samp, change_index = change_smiles_index)
        print("Time taken = ", time.time() - start_time)
    #fp = open('Generated_smiles_bay_redox_new.smi', 'w')
        with open(fname, 'w') as fp:
            for i in range(0, len(post_samples)):
                 smiles = visualize_smiles(np.array(post_samples[i]))#    
                 for j in range(0, len(smiles)):
                      st = str(smiles[j]) + "\n"
                      fp.writelines("{}".format(st))    
    except: 
        pass



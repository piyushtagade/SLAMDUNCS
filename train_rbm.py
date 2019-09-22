# -----------------------------------------------------------------------------
# Training the Restricted Boltzmann Machine on the pubchem dataset 
# -----------------------------------------------------------------------------
import scipy as sc
import numpy as np

from pprint import pprint 
# Load the dataset
from deep_learning import *
from process_smiles import * 
from pubchem_data import * 
import pickle
from anova import *
import time
from deep_neural_network import *
# -----------------------------------------------------------------------------
# Method to train the Restricted Boltzmann Machine
# -----------------------------------------------------------------------------
def train_rbm(pubchem_data_filename, max_smiles_length=100): 
# -----------------------------------------------------------------------------    
    ''' 
        Training the RBM
    '''
# -----------------------------------------------------------------------------
    smiles, properties_all = read_json_data(pubchem_data_filename, redox = False, 
                                            max_smiles_length=max_smiles_length) 

    train_set_all, test_set_all = create_smiles_dataset(smiles, properties_data=[], 
                                                        max_smiles_length=max_smiles_length, percentage_train_data = 0.3)
    training_data = np.array(train_set_all[0])
# -----------------------------------------------------------------------------
    [num_data, num_vis] = np.shape(training_data) 

    num_test_data = np.shape(test_set_all)[0]
# -----------------------------------------------------------------------------
# Creating an instance of the RBM class 
# -----------------------------------------------------------------------------                                  
    rbm = Restricted_Boltzmann_Machine(num_vis, 800)
# -----------------------------------------------------------------------------
# Setting training parameters 
# -----------------------------------------------------------------------------                                  
    numbatches = int(floor(num_data/1000)) 
    maxepoch = 3650
    learning_rate = 0.01
    cdn = 1 
    rbm.options(maxepoch, numbatches, learning_rate, cdn, method = 'Gibbs', 
                vistype='BB', initialmomentum=0.5, finalmomentum=0.9, initialepochs=5)
# -----------------------------------------------------------------------------
# Training the RBM
# -----------------------------------------------------------------------------    
    error, epochs = rbm.train(training_data, visualize=False)
# -----------------------------------------------------------------------------
# Saving the trained RBM as a pickle file for future usage    
# -----------------------------------------------------------------------------
    with open('rbm_trained.pkl', 'wb') as fp:
        pickle.dump(rbm, fp)

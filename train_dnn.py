# -------------------------------------------------------
# Training a deep neural network on properties dataset 
# -------------------------------------------------------
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

# -----------------------------------------------------------------------------
def train_dnn(pubchem_data_filename, properties_data_filename, 
              max_smiles_length=100): 
# -----------------------------------------------------------------------------
# For ensuring that same random number is always generated 
# -----------------------------------------------------------------------------
    np.random.seed(100)
# -----------------------------------------------------------------------------
# For ensuring that same random number is always generated 
# -----------------------------------------------------------------------------
    smiles, properties_all = read_json_data(pubchem_data_filename, redox = False, max_smiles_length=max_smiles_length) 

    train_set_all, test_set_all = create_smiles_dataset(smiles, properties_data=[], max_smiles_length=max_smiles_length, 
                                                        percentage_train_data = 0.3)


    smiles1, properties1 = read_json_data(properties_data_filename, redox = True, is_properties = True, 
                                          max_smiles_length=max_smiles_length) 

    smiles, properties = prune(smiles1, properties1)
    tot_data = len(smiles)
    train_set_redox, test_set_redox = create_smiles_dataset(smiles, properties_data = properties, 
                                                            max_smiles_length=max_smiles_length, percentage_train_data = 0.95)


    target_max = np.max(np.concatenate((np.array(train_set_redox[1]), np.array(test_set_redox[1])), axis=0), axis=0) 
    target_min = np.min(np.concatenate((np.array(train_set_redox[1]), np.array(test_set_redox[1])), axis=0), axis=0)


    training_data = np.concatenate((np.array(train_set_all[0]), np.array(train_set_redox[0])), axis=0)

# -----------------------------------------------------------------------------
    [num_data, num_vis] = np.shape(training_data) 


# -----------------------------------------------------------------------------
# Starting DBM
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Now pre-training the DBM
# -----------------------------------------------------------------------------
    num_test_data = np.shape(test_set_redox)[0]
                 
                  
    train_targ = (np.array(train_set_redox[1]) - target_min)/(target_max - target_min);                  
#train_targ = (np.array(train_set_redox[1]))/(target_max);                  
    training_target = train_targ 

    print("Training Target = ", np.sum(np.sum(training_target, axis = 0))) 


    num_targ = np.shape(training_target)[1]

    print(num_vis, num_targ)
    num_hid = int(np.floor(num_vis*vishid_ratio))

    [num_data, num_vis] = np.shape(training_data)  


    num_nodes = [num_vis, 800, 500, 200, 200, num_targ]


    num_layers = len(num_nodes)
    dropout = 0.7*np.ones(len(num_nodes)-1)
    dnn = Deep_Neural_Network(num_layers, num_nodes, dropout = dropout)


# -----------------------------------------------------------------------------
# Setting pretraining options 
# -----------------------------------------------------------------------------
    maxepoch = 100
    learning_rate_dbn = 0.01
    cdn_dbn = 1

    learning_rate_gb = 0.01 # This value is taken after experiments on rbm_pubchem code
    cdn_gb = 1               # This value is taken after experiments on rbm_pubchem code
 
    numbatches = int(floor(num_data/200))
#numbatches = 10  
    dnn.pretrain_options(maxepoch, numbatches, learning_rate_dbn, learning_rate_gb, cdn_dbn, cdn_gb, 
                     method = 'Gibbs', initialmomentum=0.5, finalmomentum=0.9, initialepochs=5)

# -----------------------------------------------------------------------------
# Setting training options 
# -----------------------------------------------------------------------------
    maxepoch = 10
    learning_rate = 0.1
    numbatches = int(floor(num_data/600)) 
    #numbatches = 10
    dnn.train_options(maxepoch, numbatches, learning_rate, initialmomentum=0.8, finalmomentum=0.9, initialepochs=5)

# -----------------------------------------------------------------------------
# Pretraining the DNN 
# -----------------------------------------------------------------------------

    dnn.pretrain(training_data, training_target)

    with open('dnn_pretrain.pkl', 'wb') as fp:
        pickle.dump(dnn, fp)



    smiles1, properties1 = read_json_data(properties_data_filename, redox = True, is_properties = True, 
                                          max_smiles_length=max_smiles_length) 

    smiles, properties = prune(smiles1, properties1)

#smiles, properties = prune(smiles1, properties1)

    train_set_redox, test_set_redox = create_smiles_dataset(smiles, properties_data = properties, 
                                                            max_smiles_length=max_smiles_length, 
                                                            percentage_train_data = 0.80, use_doe = False)
    training_data = np.array(train_set_redox[0]) 

    [num_data, num_vis] = np.shape(training_data) 

    train_targ = (np.array(train_set_redox[1]) - target_min)/(target_max - target_min);  
#train_targ = (np.array(train_set_redox[1]))/(target_max);
    training_target = train_targ                    

    test_data = np.array(test_set_redox[0]) 

#test_data_dbm = dbn.forward_pass(test_data)  

    test_targ = (np.array(test_set_redox[1]) - target_min)/(target_max - target_min); 
# test_targ = (np.array(test_set_redox[1]))/(target_max); 
    num_targ_data = np.shape(train_targ)[0]

    test_target = test_targ  
# ------------------------------------------------------------------------------
    epochs, errors = dnn.train(training_data, training_target)
# ------------------------------------------------------------------------------
    with open('trained_dnn.pkl', 'wb') as fp:  
        pickle.dump(dnn, fp)
# ------------------------------------------------------------------------------        
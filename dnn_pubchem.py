# -------------------------------------------------------
# Using Deep Boltzmann Machine for pubchem dataset 
# -------------------------------------------------------
import scipy as sc
import numpy as np

from pprint import pprint 
# Load the dataset
from deep_learning import *
from process_smiles import * 
from pubchem_data import * 
import pickle

smiles, properties = read_json_data('Molecular_data_pubchem') 

train_set, test_set = create_smiles_dataset(smiles, properties_data=properties, max_smiles_length=30, percentage_train_data = 0.8)

training_data = np.array(train_set[0]) 
training_target = np.array(train_set[1])/np.max(train_set[1], axis=0); 

[num_data, num_vis] = np.shape(training_data)
num_targ = np.shape(training_target)[1]

print(num_vis, num_targ)

num_layers = 4

num_nodes = [num_vis, 200, 200, num_targ]

dnn = Deep_Neural_Network(num_layers, num_nodes)

# -----------------------------------------------------------------------------
# Setting pretraining options 
# -----------------------------------------------------------------------------
maxepoch = 500
learning_rate_dbn = 0.01
cdn_dbn = 1

learning_rate_gb = 0.001 # This value is taken after experiments on rbm_pubchem code
cdn_gb = 10               # This value is taken after experiments on rbm_pubchem code
 
numbatches = int(floor(num_data/100)) 
dnn.pretrain_options(maxepoch, numbatches, learning_rate_dbn, learning_rate_gb, cdn_dbn, cdn_gb, 
                     method = 'Gibbs', initialmomentum=0.5, finalmomentum=0.9, initialepochs=5)

# -----------------------------------------------------------------------------
# Setting training options 
# -----------------------------------------------------------------------------
maxepoch = 3000  
learning_rate = 0.01
numbatches = int(floor(num_data/100)) 
numbatches = 10 
dnn.train_options(maxepoch, numbatches, learning_rate, initialmomentum=0.5, finalmomentum=0.9, initialepochs=5)

# -----------------------------------------------------------------------------
# Pre-training
# -----------------------------------------------------------------------------
#training_target[:,0] = np.sum(training_data, axis=1)
#training_target[:,1] = np.sum(training_data, axis=1)
#mx_train = np.max(training_target, axis=0)

#training_target = training_target/mx_train
#errors_dbn, epochs_dbn = 
dnn.pretrain(training_data, training_target)

with open('dnn_pretrained.pkl', 'wb') as fp:
    pickle.dump(dnn, fp)

epochs, errors = dnn.train(training_data, training_target)

with open('dnn_trained.pkl', 'wb') as fp:
    pickle.dump(dnn, fp)


test_data = np.array(test_set[0]) 
test_target = np.array(test_set[1])/np.max(train_set[1], axis=0); 


#test_target[:,0] = np.sum(test_data, axis=1)
#test_target[:,1] = np.sum(test_data, axis=1)
#test_target = test_target/mx_train



#test_target = np.array(test_set[1])/np.max(train_set[1], axis=0); 


data = training_data[0:100,:]                      
pred_target_train = dnn.predict(data)

                      
data = test_data[0:100,:]                      
pred_target_test = dnn.predict(data)


'''
with open('dbm_pretrained.pkl', 'wb') as fp:
    pickle.dump(dbm, fp)



errors_dbm, epochs_dbm, errors_vis, errors_targ = dbm.train(training_data, training_target)


test_data = np.array(test_set[0]) 
test_target = np.array(test_set[1])/np.max(train_set[1], axis=0); 


data = training_data[0:100,:]                      
pred_target_train = dbm.predict_mc(data)

                      
data = test_data[0:100,:]                      
pred_target_test = dbm.predict_mc(data)


no_samples = 100; burn_period = 10000; no_steps = 100 
visible, target = dbm.sample(no_samples, burn_period, no_steps) 

fp = open('Generated_smiles_dbm.smi', 'w')

for i in range(0, len(visible)):
    smiles = visualize_smiles(np.array(visible[i]))#
    print(len(smiles))
    for j in range(0, len(smiles)):
        st = str(smiles[j]) + "\n"
        fp.writelines("{}".format(st))    
fp.close() 
'''
#fp = open('Generated_smiles.smi', 'w')
#for i in range(0, len(smiles)):
#    st = str(smiles[i]) + "\n"
#    fp.writelines("{}".format(st))    
#fp.close()
import scipy as sc
import numpy as np
from math import floor 
from scipy import linalg
from matplotlib import pyplot as plt
import json
# -------------------------------------------------------
#                Defining the RBM
#        Builiding block for DBN and DBM
# -------------------------------------------------------
class Restricted_Boltzmann_Machine:

# Initializing the rbm 
    def __init__(self, num_vis, num_hid, weights = None, vis_bias = None, hid_bias = None):
        ''' Initializing the restricted boltzmann machine '''
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.vistype = 'BB' 
        self.weights = 0.1*np.random.random((num_vis, num_hid))   
        self.vis_bias = 0.001*np.ones(num_vis) 
        self.hid_bias = np.zeros(num_hid)
        self.variance = 1.0*np.ones(num_vis)
# ------------------------------------------------------
# Default training options
# ------------------------------------------------------
        self.cdn = 1 
        self.maxepoch = 100
        self.learning_rate = 0.05 
        self.numbatches = 1

        self.initialmomentum = 0.5
        self.finalmomentum = 0.9 
        self.initialepochs = 5 
        self.method = 'Gibbs'            
# ----------------------------------------------------------------------------
# Is the RBM already trained 
        self.is_trained = False 
# Trained for how many epochs 
        self.trained_epochs = 0  
# -----------------------------------------------------------------------------
# For Parallel Tempered Gibbs Sampling
        self.num_temps = 10 
# -----------------------------------------------------------------------------        
# -----------------------------------------------------------------------------
# For doubling of weights and biases: Used for pre-training the DBM
# -----------------------------------------------------------------------------
        self.double = False 
# -----------------------------------------------------------------------------
# For multiplying of weights and biases: Used for pre-training the DBM
# -----------------------------------------------------------------------------
        self.vis_multfact = 1.0 
        self.hid_multfact = 1.0 
# -----------------------------------------------------------------------------
# For saving error 
# -----------------------------------------------------
        self.epochs = None; self.errors = None  
# ------------------------------------------------------
# Training options
# ------------------------------------------------------
    def options(self, maxepoch, numbatches, learning_rate, cdn, method = 'Gibbs', vistype='BB', initialmomentum=0.5, finalmomentum=0.9, initialepochs=5):
# ------------------------------------------------------
        ''' Specifying the training options '''
        print(vistype)
        self.cdn = cdn 
        self.maxepoch = maxepoch
        self.learning_rate = learning_rate
        self.numbatches = numbatches
        self.vistype = vistype
        
        
        self.initialmomentum = initialmomentum
        self.finalmomentum = finalmomentum 
        self.initialepochs = initialepochs 
        self.method = method 
# ------------------------------------------------------
# Sigmoid function: used for training the RBM 
# ------------------------------------------------------
    def sigmoid(self, x): 
         ''' Sigmoid function '''
         return (1.0/(1+np.exp(-x)))
# ------------------------------------------------------
# Energy of the restricted Boltzmann machine
# ------------------------------------------------------
    def energy(self, data, weights, bias):
        ''' Function to calculate energy'''
        [num_data, num_dim] = np.shape(data)
        return np.dot(data, weights) + np.tile(bias,(num_data, 1))   
# ------------------------------------------------------
# Contrastive divergence algorithm                 
# ------------------------------------------------------
    def contrastive_divergence(self, data, visualize=False, method='Gibbs', ishalf = 1.0):
        ''' CD algorithm for training the rbm ''' 
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
        maxepoch = self.maxepoch
        numbatches = self.numbatches   
        vistype = self.vistype
        weights = self.weights
        vis_bias = self.vis_bias
        hid_bias = self.hid_bias
        variance = self.variance  
        '''if self.double == True:
            wd = 2.0
        else:
            wd = 1.0'''
        vis_multfact = self.vis_multfact
        hid_multfact = self.hid_multfact  
        print('Multfact', vis_multfact, hid_multfact)
# ------------------------------------------------------
#       Initializing the increaments  
# ------------------------------------------------------ 
        incweights = np.zeros(np.shape(weights))
        incvisbias = np.zeros(np.shape(vis_bias))
        inchidbias = np.zeros(np.shape(hid_bias))
        incvariance = np.zeros(np.shape(variance))
# CD(n) algorithm is implemented 
        cdn = self.cdn              
# Learning rate 
        learning_rate = self.learning_rate
# Momentum    
        initialepochs = self.initialepochs 
        initialmomentum = self.initialmomentum
        finalmomentum = self.finalmomentum 
        weightcost = 0.001 
        t0 = 300; eta0 = learning_rate        
# -----------------------------------------------------
# For visualization of results   
# -----------------------------------------------------
        if visualize==True:
             fig = plt.figure(2) 
             ax = fig.gca()
             fig.show()  

        if vistype == 'GB': 
             zvar = np.log(variance) 
             #sdev = np.sqrt(variance) 
# -----------------------------------------------------
        start_epoch = self.trained_epochs 

# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------
        epochs = []; errsums = []
        for epoch in range(start_epoch, maxepoch):
             errsum = 0.0
             eta = eta0/(1+(epoch/t0))  
             #eta = learning_rate 
             if eta < 0.0001:
                 eta = 0.0001 
                  
# Training each batch
             for batch in range(0, numbatches):        
# Retrive batchdata from the data              
                 batchdata = data[batch]
                 [num_data, num_dim] = np.shape(batchdata)
                 alpha = eta/num_data  
# Convert the data to binary 
                 if vistype == 'BB': 
                      batchdata = 1.0*(batchdata > np.random.random(np.shape(batchdata)))                                                                    
                 if vistype == 'GB':    
                      #print("Batchdata = ", np.sum(np.sum(batchdata, axis=0))) 
                      batchdata = batchdata/variance
                      #print("Batchdata = ", np.sum(np.sum(batchdata, axis=0))) 
                     
# Total energy to the hidden layer
                 wd = hid_multfact  
                 
                 hid_energy = self.energy(batchdata, wd*weights, wd*hid_bias)
                 '''if vistype == 'GB':                                             
                      print("Hid energy = ", np.sum(np.sum(hid_energy, axis=0)))
                      input("Hid energy")'''
# Conditional probability of the hidden layer and binary hidden states 
                 prob_hid_state = self.sigmoid(hid_energy) 
                 hid_states = 1.0*(prob_hid_state > np.random.random(np.shape(prob_hid_state)))
# Data dependent
                 if vistype == 'BB':  
                      pos_weights = np.dot(batchdata.transpose(),hid_states)
                      pos_vis_bias = np.sum(batchdata, axis=0) 
                       
                 if vistype == 'GB': 
                      pos_weights = np.dot(batchdata.transpose(),hid_states)
                      pos_vis_bias = np.sum(batchdata, axis=0)
                      bv = batchdata*variance 
                      #print(np.shape(weights), np.shape(hid_states))
                      var1 = 0.5*(bv - np.tile(vis_bias, (num_data, 1)))**2   
                      var2 = bv * (np.dot(weights, hid_states.T).T)
                      pos_var = np.sum((var1 - var2), axis=0)  
                 pos_hid_bias = np.sum(hid_states, axis=0)
# Starting the negative phase
                 if method == 'Gibbs': 
# Using alternate Gibbs sampling                       
                     list_vis, list_hid = self.gibbs(batchdata, weights, vis_bias, hid_bias, vis_multfact, hid_multfact, 
                                                     total_samps = cdn, initialize_vis = True, vistype=vistype, variance = variance)                     
                     negdata = list_vis[cdn-1]
                 else:
                     nt = self.num_temps 
                     negdata = self.tempered(batchdata, weights, vis_bias, hid_bias, num_temps = nt, initialize_vis = True)[0][0] 
  
                 if vistype == 'GB': 
                     negdata = negdata/variance      
 
# Total energy to the hidden layer
                 #wd = hid_multfact
                 hid_energy = self.energy(negdata, wd*weights, wd*hid_bias)
# Conditional probability of the hidden layer and binary hidden states 
                 prob_hid_state = self.sigmoid(hid_energy) 
                 hid_states = 1.0*(prob_hid_state > np.random.random(np.shape(prob_hid_state)))

# -----------------------------------------------------------------------------
# Reconstruction error
# -----------------------------------------------------------------------------
                 
# Model dependent
                 if vistype == 'BB': 
                     neg_weights = np.dot(negdata.T, hid_states)
                     neg_vis_bias = np.sum(negdata, axis=0)
                     errsum = errsum + np.sum((batchdata - negdata)**2.0) #/(num_data*num_dim)


                 if vistype == 'GB': 
                     errsum = errsum + np.sum((batchdata*variance - negdata*variance)**2.0) #/(num_data*num_dim)
                     neg_weights = np.dot(negdata.T,hid_states)
                     neg_vis_bias = np.sum(negdata, axis=0)
                     bv = negdata*variance 
                     var1 = 0.5*(bv - np.tile(vis_bias, (num_data, 1)))**2   
                     var2 = bv*(np.dot(weights, hid_states.T).T) 
                     neg_var = np.sum((var1 - var2), axis=0) 
                     
                 neg_hid_bias = np.sum(hid_states, axis=0)
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 incweights = momentum*incweights + alpha*(pos_weights - neg_weights) - alpha*num_data*weightcost*incweights 
                 incvisbias = momentum*incvisbias + alpha*(pos_vis_bias - neg_vis_bias)
                 inchidbias = momentum*inchidbias + alpha*(pos_hid_bias - neg_hid_bias) 

                 if vistype == 'GB': 
                     incvariance = momentum*incvariance + 1e-03*np.exp(-zvar)*alpha*(pos_var - neg_var)                      
                     
# Updating the parameters 
                 weights  = weights  +  incweights
                 vis_bias = vis_bias +  incvisbias
                 hid_bias = hid_bias +  inchidbias
                 
                 if vistype == 'GB': 
                     if epoch > 100:
                         zvar = zvar + incvariance
#zvar = zvar + incvariance
                     variance = np.exp(zvar)
                     variance[np.where(variance < 1e-05)] = 1e-05
                     zvar = np.log(variance)                     
                                                
             if epoch%2 == 0: 
                 print('Epoch = ', epoch, 'Error = ',errsum)
             epochs.append(epoch); errsums.append(errsum)
# -----------------------------------------------------
# Updated parameters 
        self.weights  = weights
        self.vis_bias = vis_bias 
        self.hid_bias = hid_bias 
        self.variance = variance
        self.epochs = epochs 
        self.errors = errsums 
        self.is_trained = True 
        self.trained_epochs = maxepoch        
 
        return errsums, epochs
# ------------------------------------------------------
# Contrastive divergence algorithm                 
# ------------------------------------------------------
    def contrastive_divergence_adadelta(self, data, visualize=False, method='Gibbs', ishalf = 1.0):
        ''' CD algorithm for training the rbm ''' 
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
        maxepoch = self.maxepoch
        numbatches = self.numbatches   
        vistype = self.vistype
        weights = self.weights
        vis_bias = self.vis_bias
        hid_bias = self.hid_bias
        variance = self.variance  
        '''if self.double == True:
            wd = 2.0
        else:
            wd = 1.0'''
        vis_multfact = self.vis_multfact
        hid_multfact = self.hid_multfact  
        print('Multfact', vis_multfact, hid_multfact)
# ------------------------------------------------------
#       Initializing the increaments  
# ------------------------------------------------------ 
        incweights = np.zeros(np.shape(weights))
        incvisbias = np.zeros(np.shape(vis_bias))
        inchidbias = np.zeros(np.shape(hid_bias))
        incvariance = np.zeros(np.shape(variance))

# ------------------------------------------------------
#       Initializing gradient moments
# ------------------------------------------------------ 
        gmweights = np.zeros(np.shape(weights))
        gmvisbias = np.zeros(np.shape(vis_bias))
        gmhidbias = np.zeros(np.shape(hid_bias))
        gmvariance = np.zeros(np.shape(variance))        
        
# ------------------------------------------------------
#       Initializing increament moments
# ------------------------------------------------------ 
        imweights = np.zeros(np.shape(weights))
        imvisbias = np.zeros(np.shape(vis_bias))
        imhidbias = np.zeros(np.shape(hid_bias))
        imvariance = np.zeros(np.shape(variance))                
# ------------------------------------------------------                
# CD(n) algorithm is implemented 
        cdn = self.cdn              
# Learning rate 
        learning_rate = self.learning_rate
# Momentum    
        initialepochs = self.initialepochs 
        initialmomentum = self.initialmomentum
        finalmomentum = self.finalmomentum 
        weightcost = 0.001 
        t0 = 300; eta0 = learning_rate  
        eps = 1e-08 
# -----------------------------------------------------
# For visualization of results   
# -----------------------------------------------------
        if visualize==True:
             fig = plt.figure(2) 
             ax = fig.gca()
             fig.show()  

        if vistype == 'GB': 
             zvar = np.log(variance) 
             #sdev = np.sqrt(variance) 
# -----------------------------------------------------
        start_epoch = self.trained_epochs 

# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------
        epochs = []; errsums = []
        for epoch in range(start_epoch, maxepoch):
             errsum = 0.0
             eta = eta0/(1+(epoch/t0))  
             #eta = learning_rate 
             if eta < 0.0001:
                 eta = 0.0001 
                  
# Training each batch
             for batch in range(0, numbatches):        
# Retrive batchdata from the data              
                 batchdata = data[batch]
                 [num_data, num_dim] = np.shape(batchdata)
                 alpha = eta/num_data  
# Convert the data to binary 
                 if vistype == 'BB': 
                      batchdata = 1.0*(batchdata > np.random.random(np.shape(batchdata)))                                                                    
                 if vistype == 'GB':    
                      #print("Batchdata = ", np.sum(np.sum(batchdata, axis=0))) 
                      batchdata = batchdata/variance
                      #print("Batchdata = ", np.sum(np.sum(batchdata, axis=0))) 
                     
# Total energy to the hidden layer
                 wd = hid_multfact  
                 
                 hid_energy = self.energy(batchdata, wd*weights, wd*hid_bias)
                 '''if vistype == 'GB':                                             
                      print("Hid energy = ", np.sum(np.sum(hid_energy, axis=0)))
                      input("Hid energy")'''
# Conditional probability of the hidden layer and binary hidden states 
                 prob_hid_state = self.sigmoid(hid_energy) 
                 hid_states = 1.0*(prob_hid_state > np.random.random(np.shape(prob_hid_state)))
# Data dependent
                 if vistype == 'BB':  
                      pos_weights = np.dot(batchdata.transpose(),hid_states)
                      pos_vis_bias = np.sum(batchdata, axis=0) 
                       
                 if vistype == 'GB': 
                      pos_weights = np.dot(batchdata.transpose(),hid_states)
                      pos_vis_bias = np.sum(batchdata, axis=0)
                      bv = batchdata*variance 
                      #print(np.shape(weights), np.shape(hid_states))
                      var1 = 0.5*(bv - np.tile(vis_bias, (num_data, 1)))**2   
                      var2 = bv * (np.dot(weights, hid_states.T).T)
                      pos_var = np.sum((var1 - var2), axis=0)  
                 pos_hid_bias = np.sum(hid_states, axis=0)
# Starting the negative phase
                 if method == 'Gibbs': 
# Using alternate Gibbs sampling                       
                     list_vis, list_hid = self.gibbs(batchdata, weights, vis_bias, hid_bias, vis_multfact, hid_multfact, 
                                                     total_samps = cdn, initialize_vis = True, vistype=vistype, variance = variance)                     
                     negdata = list_vis[cdn-1]
                 else:
                     nt = self.num_temps 
                     negdata = self.tempered(batchdata, weights, vis_bias, hid_bias, num_temps = nt, initialize_vis = True)[0][0] 
  
                 if vistype == 'GB': 
                     negdata = negdata/variance      
 
# Total energy to the hidden layer
                 #wd = hid_multfact
                 hid_energy = self.energy(negdata, wd*weights, wd*hid_bias)
# Conditional probability of the hidden layer and binary hidden states 
                 prob_hid_state = self.sigmoid(hid_energy) 
                 hid_states = 1.0*(prob_hid_state > np.random.random(np.shape(prob_hid_state)))

# -----------------------------------------------------------------------------
# Reconstruction error
# -----------------------------------------------------------------------------
                 
# Model dependent
                 if vistype == 'BB': 
                     neg_weights = np.dot(negdata.T, hid_states)
                     neg_vis_bias = np.sum(negdata, axis=0)
                     errsum = errsum + np.sum((batchdata - negdata)**2.0) #/(num_data*num_dim)


                 if vistype == 'GB': 
                     errsum = errsum + np.sum((batchdata*variance - negdata*variance)**2.0) #/(num_data*num_dim)
                     neg_weights = np.dot(negdata.T,hid_states)
                     neg_vis_bias = np.sum(negdata, axis=0)
                     bv = negdata*variance 
                     var1 = 0.5*(bv - np.tile(vis_bias, (num_data, 1)))**2   
                     var2 = bv*(np.dot(weights, hid_states.T).T) 
                     neg_var = np.sum((var1 - var2), axis=0) 
                     
                 neg_hid_bias = np.sum(hid_states, axis=0)
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# -----------------------------------------------------------------------------                     
#  Obtaining gradient moments 
# -----------------------------------------------------------------------------
                 gmweights = momentum*gmweights + (1.0-momentum)*(pos_weights - neg_weights)*(pos_weights - neg_weights)
                 gmvisbias = momentum*gmvisbias + (1.0-momentum)*(pos_vis_bias - neg_vis_bias)*(pos_vis_bias - neg_vis_bias)                                 
                 gmhidbias = momentum*gmhidbias + (1.0-momentum)*(pos_hid_bias - neg_hid_bias)*(pos_hid_bias - neg_hid_bias)                                                  
# -----------------------------------------------------------------------------                     
#  Obtaining inreament moments 
# -----------------------------------------------------------------------------                 
                 imweights = momentum*imweights + (1.0-momentum)*(incweights)*(incweights)
                 imvisbias = momentum*imvisbias + (1.0-momentum)*(incvisbias)*(incvisbias)
                 imhidbias = momentum*imhidbias + (1.0-momentum)*(inchidbias)*(inchidbias)
                 
# -----------------------------------------------------------------------------
# Obtaining increaments 
# -----------------------------------------------------------------------------
                 alpha = (np.sqrt(imweights + eps))/(np.sqrt(gmweights + eps))
                 incweights = alpha*(pos_weights - neg_weights)
# Updating the parameters                  
                 weights  = weights  +  incweights - alpha*weightcost*incweights
                 
                 alpha = (np.sqrt(imvisbias + eps))/(np.sqrt(gmvisbias + eps))
                 incvisbias = alpha*(pos_vis_bias - neg_vis_bias)
                 vis_bias = vis_bias +  incvisbias
                 alpha = (np.sqrt(imhidbias + eps))/(np.sqrt(gmhidbias + eps))
                 inchidbias = alpha*(pos_hid_bias - neg_hid_bias)                 
                 hid_bias = hid_bias +  inchidbias
                                      
                 if vistype == 'GB': 
                     incvariance = momentum*incvariance + 1e-03*np.exp(-zvar)*alpha*(pos_var - neg_var)                                           
                 
                 if vistype == 'GB': 
                     if epoch > 100:
                         zvar = zvar + incvariance
#zvar = zvar + incvariance
                     variance = np.exp(zvar)
                     variance[np.where(variance < 1e-05)] = 1e-05
                     zvar = np.log(variance)                     
                                                
             if epoch%2 == 0: 
                 print('Epoch = ', epoch, 'Error = ',errsum)
             epochs.append(epoch); errsums.append(errsum)
# -----------------------------------------------------
# Updated parameters 
        self.weights  = weights
        self.vis_bias = vis_bias 
        self.hid_bias = hid_bias 
        self.variance = variance
        self.epochs = epochs 
        self.errors = errsums 
        self.is_trained = True 
        self.trained_epochs = maxepoch        
 
        return errsums, epochs
# -----------------------------------------------------    
    
# -----------------------------------------------------
# Train the RBM  
# -----------------------------------------------------
    def train(self, data, visualize=False, method='Gibbs', ishalf = 1.0):
        ''' Training the RBM '''
        numbatches = self.numbatches 
        [num_data, num_dim] = np.shape(data) 
        
        num_batch_data = int(floor(num_data/numbatches))
        num_last_batch_data = num_data - (numbatches-1)*num_batch_data 

        perm_data = np.random.permutation(num_data) 
        
        list_data = [] 

        for batch in range(0, numbatches-1):
            indx_batch = perm_data[(batch)*num_batch_data:(batch+1)*num_batch_data]
            list_data.append(data[indx_batch, :])   
            #print("Indx batch data = ", np.sum(np.sum(indx_batch, axis=0)), batch)

        indx_batch = perm_data[(numbatches-1)*num_batch_data:num_data]
         
        list_data.append(data[indx_batch, :])         
# ------------------------------------------------------
        error, epochs = self.contrastive_divergence_adadelta(list_data, visualize, method=method, ishalf = ishalf)
# ------------------------------------------------------        
        return error, epochs
# ------------------------------------------------------
# Visible to hidden 
# ------------------------------------------------------
    def visible_to_hidden(self, vis):
        ''' Function for visible to hidden '''
        vis = 1.0*(vis > np.random.random(np.shape(vis)))
        return self.sigmoid(self.energy(vis, self.weights, self.hid_bias)) 


# ------------------------------------------------------
# Hidden to visible 
# ------------------------------------------------------
    def hidden_to_visible(self, hid):
        ''' Function for hidden to visible'''
        hid = 1.0*(hid > np.random.random(np.shape(hid)))
        return self.sigmoid(self.energy(hid, self.weights.transpose(), self.vis_bias))


# ------------------------------------------------------
# Energy of RBM  
# ------------------------------------------------------
    def rbm_energy(self, vis, hid, w, b, c):
        ''' Function for energy of RBM '''
        return -np.diag(np.dot(vis, np.dot(w, hid.transpose()))) - np.dot(vis, b) - np.dot(hid, c)


# -----------------------------------------------------------------------------
# Function for calculating log of probability ratio between two random vis 
# -----------------------------------------------------------------------------
    def visible_prob_ratio(self, vis1, vis2, weights, vis_bias, hid_bias):
        ''' Function for calculating log of probability ratio between two random vis
            log (p(vis1)/p(vis2)) is returned'''
        vis1 = 1.0*(vis1 > np.random.random(np.shape(vis1)))
        vis2 = 1.0*(vis2 > np.random.random(np.shape(vis2))) 

        vis_dim= np.shape(vis_bias)[0]
        if vis1.ndim == 1:
            vis1 = vis1.reshape(1, vis_dim); 
        if vis2.ndim == 1:            
            vis2 = vis2.reshape(1, vis_dim);                                            
        num_data = np.shape(vis1)[0] 
# -----------------------------------------------------------------------------        
# bias term
# -----------------------------------------------------------------------------                
        bv = np.dot(vis1-vis2, (vis_bias.reshape(vis_dim, 1)))
        bv = bv[:,0]
# -----------------------------------------------------------------------------        
# Calculating the energies for vis1 and vis2          
# -----------------------------------------------------------------------------        
        e1 = self.energy(vis1, weights, hid_bias)
        e2 = self.energy(vis2, weights, hid_bias)
# -----------------------------------------------------------------------------        
# Sigmoides for vis1 and vis2 
# -----------------------------------------------------------------------------        
        sigm1 = self.sigmoid(e1); sigm1[np.where(sigm1 < 1e-100)] = 1e-100 
        sigm2 = self.sigmoid(e2); sigm2[np.where(sigm2 < 1e-100)] = 1e-100 
# -----------------------------------------------------------------------------        
# Conditioning over hid terms
# -----------------------------------------------------------------------------
        cond = np.sum((e1-e2) + np.log(sigm2) - np.log(sigm1), axis = 1);
        return bv + cond
# -----------------------------------------------------------------------------
# Function for calculating log of probability of visible node
# -----------------------------------------------------------------------------
    def visible_probability(self, vis, weights, vis_bias, hid_bias):
        ''' Function for calculating log of probability log (p(vis1)) '''
        vis = 1.0*(vis > np.random.random(np.shape(vis)))
        vis_dim= np.shape(vis_bias)[0]
        if vis.ndim == 1:
            num_data = 1
            vis = vis.reshape(1, vis_dim); 
# -----------------------------------------------------------------------------        
# bias term
# -----------------------------------------------------------------------------                
        bv = np.dot(vis, (vis_bias.reshape(vis_dim, 1)))
# -----------------------------------------------------------------------------        
# Calculating the energies for vis1 and vis2          
# -----------------------------------------------------------------------------        
        e1 = self.energy(vis, weights, hid_bias)
# -----------------------------------------------------------------------------        
# Sigmoides for vis1 and vis2 
# -----------------------------------------------------------------------------        
        sigm1 = self.sigmoid(e1); sigm1[np.where(sigm1 < 1e-100)] = 1e-100 
# -----------------------------------------------------------------------------        
# Conditioning over hid terms
# -----------------------------------------------------------------------------
        cond = np.sum((e1) - np.log(sigm1), axis = 1); 
        return bv + cond

# -----------------------------------------------------------------------------        
# Proposal distribution based on RBM
# Returns proposed state, pres to prop probability and prop to pres probability
# -----------------------------------------------------------------------------
    def rbm_proposal(self, vis, change_indx, copy_loc, weights, vis_bias, hid_bias):
        ''' Proposal distribution based on RBM ''' 
# -----------------------------------------------------------------------------
        if vis.ndim == 1:            
            vis = vis.reshape(1, vis_dim)

        num_data, vis_dim = np.shape(vis)
        vis = 1.0*(vis > np.random.random(np.shape(vis)))
# -----------------------------------------------------------------------------
        idx_change_data = []; #np.zeros((num_data, num_change*8), dtype=int) 
# -----------------------------------------------------------------------------
# Selecting indices to change
# -----------------------------------------------------------------------------
        for jd in range(0, num_data): 
            idx_change = []
            idx_copy = [] 
            change_dim = min(np.shape(change_indx[jd])[0], 5)    
            if change_dim == 0:
                change_dim = 1 
            num_change = np.random.choice(change_dim, 1, replace=False)[0]
                 
            if num_change == 0:
                num_change =  1
            #num_change = 3  
            #print(change_indx[jd][-1], num_change)
            if change_indx[jd][-1]-num_change > 0:
                 ini_change = np.random.choice(change_indx[jd][-1]-num_change, 1, replace=False)    
            else:
                 ini_change = 0
            select_change = np.intersect1d(np.arange(ini_change, ini_change+num_change, 1, dtype=np.int32), change_indx[jd])
            
            #print(select_change)
            #print('num_change = ',num_change, 'select_change = ', select_change, 'change_dim = ', change_dim)        
            for i in range(0, len(select_change)):
                for j in range(0, 8):                
                    idx_change.append(select_change[i]*8 + j)         
# -----------------------------------------------------------------------------        
            idx_change_data.append(idx_change)
# -----------------------------------------------------------------------------
# Obtaining hidden state from present visible state         
        prob_hid = self.sigmoid(self.energy(vis, weights, hid_bias)) 
        hid = 1.0*(prob_hid > np.random.random(np.shape(prob_hid)))
# -----------------------------------------------------------------------------
# Obtaining proposed visisble state from the hidden state                 
        prob_vis = self.sigmoid(self.energy(hid, weights.transpose(), vis_bias))
        vis1 = 1.0*(prob_vis > np.random.random(np.shape(prob_vis)))
         
        for ic in range(0, len(copy_loc)):
            c1 = copy_loc[ic][0]; c2 = copy_loc[ic][1]
            
            cp1 = []; cp2 = []  
            for i in range(0, len(c1)):
                for j in range(0, 8):                
                    cp1.append(c1[i]*8 + j)
                    cp2.append(c2[i]*8 + j)
            if np.random.random(1) > 0.5:
                idx_copy.append([cp2, cp1])
                #vis1[:,cp2] = vis1[:,cp1]
            else:
                #vis1[:,cp1] = vis1[:,cp2]
                idx_copy.append([cp1, cp2])
        
        prop_vis = vis 

        for jd in range(0, num_data):
            idx = idx_change_data[jd]  
            #print(idx) 
            prop_vis[jd,idx] = vis1[jd,idx]
            
            for ic in range(0, len(idx_copy)):
                 cp1 = idx_copy[ic][0]; cp2 = idx_copy[ic][1]
                 #print(cp2, np.shape(prop_vis)) 
                 if cp2[-1] < np.shape(prop_vis)[1] and cp1[-1] < np.shape(prop_vis)[1]:                    
                    prop_vis[jd,cp1] = prop_vis[jd,cp2]
            
            #prop_vis[jd,:] = vis1[jd,:]
# -----------------------------------------------------------------------------
# Obtaining transition probabilites from present to proposed 
        prob_hs_v  = hid*prob_hid + (1.0-hid)*(1.0-prob_hid) # p(h*|v)
        prob_hs_v[np.where(prob_hs_v < 1e-100)] = 1e-100
        log_hs_v = np.log(prob_hs_v)          
        prob_vs_hs = vis*prob_vis + (1.0-vis)*(1.0-prob_vis) # p(v*|h*)                          
        prob_vs_hs[np.where(prob_vs_hs < 1e-100)] = 1e-100
        log_vs_hs = np.log(prob_vs_hs)            
# -----------------------------------------------------------------------------        
# Obtaining transition probabilites from proposed to present 
        prob_hid = self.sigmoid(self.energy(prop_vis, weights, hid_bias))        
        prob_hs_vs =  hid*prob_hid + (1.0-hid)*(1.0-prob_hid) # p(h*|v)
        prob_hs_vs[np.where(prob_hs_vs < 1e-100)] = 1e-100
        log_hs_vs = np.log(prob_hs_vs)
        prob_v_hs  = prop_vis*prob_vis + (1.0-prop_vis)*(1.0-prob_vis)  #p(v|h*)
        prob_v_hs[np.where(prob_v_hs < 1e-100)] = 1e-100
        log_v_hs = np.log(prob_v_hs)
        
# -----------------------------------------------------------------------------        
# Obtaining log of transition probabilites ratio 
        
        log_hid_prob_ratio = np.sum(log_hs_v  - log_hs_vs, axis=1)
        log_vis_prob_ratio = np.sum(log_vs_hs - log_v_hs, axis=1)
                         
        log_prop_ratio = log_hid_prob_ratio + log_vis_prob_ratio
        
        return prop_vis, log_prop_ratio
# -----------------------------------------------------------------------------

# -------------------------------------------------------
# Function for Gibbs sampling 
# -------------------------------------------------------
    def gibbs(self, ini_samp, weights, vis_bias, hid_bias, vis_mf, hid_mf, burn_period = 0, total_samps = 1, initialize_vis=False, vistype='BB', variance=1.0):
        ''' Function for Gibbs sampling ''' 
        if initialize_vis == True:
             if vistype == 'BB':
                  vis = 1.0*(ini_samp > np.random.random(np.shape(ini_samp)))
             else:
                  vis = ini_samp
           
                  
                  
             prob_hid = self.sigmoid(self.energy(vis, hid_mf*weights, hid_mf*hid_bias))
             hid = 1.0*(prob_hid > np.random.random(np.shape(prob_hid)))              
        else: 
             hid = ini_samp   
        list_vis = []; list_hid = [] 
        for steps in range(0, burn_period+total_samps):            
             if vistype == 'BB': 
                  prob_vis = self.sigmoid(self.energy(hid, (vis_mf*weights).transpose(), vis_mf*vis_bias))
                  vis = 1.0*(prob_vis > np.random.random(np.shape(prob_vis))) 
             if vistype == 'GB':                  
                  vis_var = self.gb_normal(hid, (vis_mf*weights).transpose(), vis_mf*vis_bias, variance)             
                  vis = vis_var/variance 

             prob_hid = self.sigmoid(self.energy(vis, hid_mf*weights, hid_mf*hid_bias))
             hid = 1.0*(prob_hid > np.random.random(np.shape(prob_hid))) 
             

             if steps >= burn_period:
                 if vistype == 'BB':
                     list_vis.append(vis)
                 if vistype == 'GB':
                     list_vis.append(vis*variance)
                 list_hid.append(hid)   
# 
        return list_vis, list_hid    

# -------------------------------------------------------
# Function for Parallel Tempered Gibbs sampling 
# -------------------------------------------------------
    def tempered(self, ini_samp, weights, vis_bias, hid_bias, num_temps = 10, num_steps = 1, total_samps = 1, initialize_vis = True): 
        ''' Function for Parallel Tempered Gibbs sampling ''' 
        list_vis = []; list_hid = []
        num_data = np.shape(ini_samp)[0]
        temperatures = np.linspace(0.0, 1.0, num_temps)   
         
        for samps in range(0, total_samps): 
# First sampling at (inverted) temp = 0 
             temps = temperatures[0]
             [v1, h1] = self.gibbs(ini_samp, temps*weights, temps*vis_bias, temps*hid_bias, initialize_vis = initialize_vis)
             v1 = v1[0]; h1 = h1[0]
             e1 = self.rbm_energy(v1, h1, weights, vis_bias, hid_bias)
# Collecting tempered samples
             for i in range(1, num_temps):
                  temps = temperatures[i]
                  [v2, h2] = self.gibbs(ini_samp, temps*weights, temps*vis_bias, temps*hid_bias, initialize_vis = initialize_vis)
                  v2 = v2[0]; h2 = h2[0]
                  e2 = self.rbm_energy(v2, h2, weights, vis_bias, hid_bias)
                  t1 = temperatures[i-1]; t2 = temperatures[i] 
                  e11 = t1*e1; e12 = t2*e1; e21 = t1*e2; e22 = t2*e2 
# Swapping
                  p = np.exp(e11 + e22 - e12 - e21)
                  r = 1.0*(p > np.random.random(np.shape(p)))
                  v2[r==1] = v1[r==1]
                  h2[r==1] = h1[r==1] 
                  e2[r==1] = e1[r==1]
# Restarting                         
                  e1 = e2; v1 = v2; h1 = h2  


             list_vis.append(v2) 
             list_hid.append(h2)  

        return [list_vis, list_hid] 
# ---------------------------------------------------------------------------
# Function for sampling visible layer from Gauss-Bernoulli RBM 
# ---------------------------------------------------------------------------
    def gb_normal(self, hid, weights, bias, variance):
        ''' Sampling from visible layer of GB RBM '''

        nd = np.shape(hid)[0]
        vis = np.dot(hid, weights) + np.tile(bias,(nd, 1))
        return vis  


    def toJSON(self):
        weights = self.weights; hid_bias = self.hid_bias; vis_bias = self.vis_bias
        variance = self.variance
        json_data = {}; json_data['weights'] = weights.tolist(); 
        json_data['hid_bias'] = hid_bias.tolist(); 
        json_data['vis_bias'] = vis_bias.tolist();  
        json_data['variance'] = variance.tolist();          
        return json.dumps(json_data, sort_keys = True, indent = 4)

# ---------------------------------------------------------------------------
# Function for saving the RBM
# ---------------------------------------------------------------------------
    def save(self, fname): 
        rbm_json = self.toJSON()
        with open(fname, 'w') as outfile:
            json.dump(rbm_json, outfile)

# ------------------------------------------------------
# Function to sample from a rbm 
# ------------------------------------------------------
    def sample_new(self, no_samples, burn_period, no_steps):
        ''' Function for sampling from RBM using alternating Gibbs sampling. 
        At present implemented only for BB RBM'''
# Number of hidden units        
        num_hid = self.num_hid
        num_vis = self.num_vis
        prob_hid = np.random.random([no_samples,num_hid])
        hid = 1.0*(prob_hid > np.random.random(np.shape(prob_hid)))        
        weights = self.weights
        vis_bias = self.vis_bias
        hid_bias = self.hid_bias
        list_vis = []; list_hid = []; 
        for steps in range(0, burn_period+no_steps):            
#            vis = self.gibbs(hid, weights, vis_bias, hid_bias, burn_period=1, total_samps = no_samples, initialize_vis=False)[0][0]                                     
            vis = self.tempered(hid, weights, vis_bias, hid_bias, num_temps = 20, initialize_vis = False)[0][0]            
# Total energy to the hidden layer
            hid_energy = self.energy(vis, weights, hid_bias)
# Conditional probability of the hidden layer and binary hidden states 
            prob_hid = self.sigmoid(hid_energy) 
            hid = 1.0*(prob_hid > np.random.random(np.shape(prob_hid)))
            if steps >= burn_period:
                 list_vis.append(vis)                 
                 list_hid.append(hid)             
        
        return list_vis      
  
    
#------------------------------------------------------------------------------    
    def sample(self, no_samples, burn_period):
        ''' Function for sampling from RBM using alternating Gibbs sampling'''
# Number of hidden units        
        num_hid = self.num_hid
        num_vis = self.num_vis
        prob_hid = np.random.random([no_samples,num_hid])
        hid = 1.0*(prob_hid > np.random.random(np.shape(prob_hid)))
        samps = np.zeros([no_samples, num_vis])
        isSampling = True; isBurnout = False; iSamp = 0
        weights = self.weights
        vis_bias = self.vis_bias
        hid_bias = self.hid_bias
        for isamp in range(0, burn_period+1):
            print(isamp)
            vis_energy = self.energy(hid, weights.transpose(), vis_bias)
            prob_vis = self.sigmoid(vis_energy)
            vis = 1.0*(prob_vis > np.random.random(np.shape(prob_vis)))     
# Total energy to the hidden layer
            hid_energy = self.energy(vis, weights, hid_bias)
# Conditional probability of the hidden layer and binary hidden states 
            prob_hid = self.sigmoid(hid_energy) 
            hid = 1.0*(prob_hid > np.random.random(np.shape(prob_hid)))
                        
        
        return vis    
# ----------------------------------------------------------------------------    
# Function to sample from a conditional RBM
# A portion of the visible layer is fixed
# ----------------------------------------------------------------------------    
    def conditional(self, vis, fixed_indx):
        ''' A function for sampling from a conditional RBM.
            As the conditional probability distribution of the 
            RBM is highly complex, MCMC is used to sample from
            the conditional RBM. 
            RBM with artificially annealed energy is used as a 
            proposal distribution. 
            The proposal distribution is defined as 
            q(v*, v) = p(h*|v) p(v*|h*)
        '''    
# -----------------------------------------------------------------------------
        weights   =  self.weights
        hid_bias  =  self.hid_bias
        vis_bias  =  self.vis_bias
# -----------------------------------------------------------------------------    
        vis_dim  = np.shape(vis_bias)[0] # dimension of the visible layer
        pres_vis = np.zeros(vis_dim); #prop_vis = pres_vis
# -----------------------------------------------------------------------------            
# Creating present state 
# -----------------------------------------------------------------------------            
        pres_vis = vis
# ----------------------------------------------------
# For collecting the samples 
        post_samples = []; accepted_samples = []  
# ----------------------------------------------------          
# Initializing the MCMC sampling 
# Number of samples 
        number_of_samples = 30000
        burnout_period = 30000
# ----------------------------------------------------
# Starting MCMC sampling  
        accepted = 0; is_accepted = False          
        for samp in range(0, burnout_period+number_of_samples):
            beta =0.7*np.random.random(1) # high temperature to allow better mixing  
            #beta = 0.5
            prop_vis, prop_ratio = self.rbm_proposal(pres_vis, fixed_indx, beta*weights, beta*vis_bias, beta*hid_bias)
            like_ratio = self.visible_prob_ratio(prop_vis, pres_vis, weights, vis_bias, hid_bias)
# Acceptance probability 
            acceptance_prob = like_ratio + prop_ratio

# Metropolis-Hastings criterion
            urand = np.random.random(1)
            
            if np.exp(acceptance_prob) > urand:           
# Proposed state accepted 
                accepted = accepted + 1
# --------------------------------------------------------                  
                pres_vis = prop_vis
                
                state = prop_vis 
                is_accepted = True
            else:  
# --------------------------------------------------------                  
                pres_vis = pres_vis
                
                state = pres_vis
                is_accepted = False
                
            '''print(urand, np.exp(acceptance_prob), is_accepted, like_ratio, prop_ratio)
            input()''' 
# ---------------------------------------------------------
# Collecting samples after burnout period 
# ---------------------------------------------------------
            if samp > burnout_period:
                post_samples.append(state) 
                if is_accepted == True:
                    accepted_samples.append(state)

            '''print(state[0, 152:160])
            print(pres_vis[0, 152:160])
            print(prop_vis[0, 152:160])'''
# ---------------------------------------------------------
# Keeping count
# ---------------------------------------------------------
            if np.mod(samp, 1000) == 0:
                print(samp, accepted) 

 
        return post_samples, accepted_samples    
  

# -----------------------------------------------------------------------------
# Function for proposal distribution. Primary purpose of this function is to 
# connect with the Deep Bayesian Inference class
# -----------------------------------------------------------------------------
    def proposal(self, vis, fwd_temp, back_temp, change_index = [], copy_loc = []):
        ''' Function for proposal distribution. 
            Primary purpose is to communicate with the Deep Bayesian Inference
            class ''' 
                

        weights   =  self.weights
        hid_bias  =  self.hid_bias
        vis_bias  =  self.vis_bias
# -----------------------------------------------------------------------------
        prop_vis, prop_ratio = self.rbm_proposal(vis, change_index, copy_loc,
                                                 fwd_temp*weights, vis_bias, hid_bias)
        prior_ratio = self.visible_prob_ratio(prop_vis, vis, weights, 
                                             vis_bias, hid_bias)

        #smiles = visualize_smiles(np.array(prop_vis))
        #print(smiles)
        return prop_vis, prop_ratio, prior_ratio
# ----------------------------------------------------------------------------
# Class for deep belief network
# ----------------------------------------------------------------------------
class Deep_Belief_Network: 
    ''' Class for defining and training the deep belief network '''  
    def __init__(self, num_layers, num_nodes):
        ''' Function to initialize the DBN '''  

        self.num_layers = num_layers 
        self.numrbms = num_layers - 1
        self.double = False
# Initializing the rbms 
        rbms = []
        for i in range(0, num_layers-1):
            rbms.append(Restricted_Boltzmann_Machine(num_nodes[i], num_nodes[i+1]))
# ------------------------------------------------------
# Default training options
# ------------------------------------------------------
            rbms[i].cdn = 1 
            rbms[i].maxepoch = 100
            rbms[i].learning_rate = 0.01
            rbms[i].numbatches = 1
            rbms[i].initialmomentum = 0.5
            rbms[i].finalmomentum = 0.9 
            rbms[i].initialepochs = 5 
            rbms[i].method = 'Gibbs'
            rbms[i].vistype = 'BB'
            if self.double == True:
                rbms[i].double = True
        self.rbms = rbms

# ------------------------------------------------------
# Training options
# ------------------------------------------------------
    def options(self, maxepoch, numbatches, learning_rate, cdn, method = 'Gibbs', vistype = 'BB', initialmomentum=0.5, finalmomentum=0.9, initialepochs=5):
# ------------------------------------------------------
        ''' Specifying the training options '''
        numrbms = self.numrbms
        for rbm in range(0, numrbms):
            self.rbms[rbm].cdn = cdn 
            self.rbms[rbm].maxepoch = maxepoch
            self.rbms[rbm].learning_rate = learning_rate
            self.rbms[rbm].numbatches = numbatches
            self.rbms[rbm].method = method            
            self.rbms[rbm].initialmomentum = initialmomentum
            self.rbms[rbm].finalmomentum = finalmomentum
            self.rbms[rbm].initialepochs = initialepochs 
            self.rbms[rbm].vistype = 'BB'
        print(vistype) 
        self.rbms[0].vistype = vistype     

# ------------------------------------------------------
# Training the deep belief network
# ------------------------------------------------------
    def train(self, data, visualize=False, ishalf = 1.0):
        ''' Function to train the deep belief belief network ''' 
        visdata = data
        #visdata = 1.0*(visdata > np.random.random(np.shape(visdata)))
        numrbms = self.numrbms 
        epochs = []; errors = [] 
        mult_fact = 1.0 
        for rbm in range(0, numrbms):
            method = self.rbms[rbm].method
            if ishalf == 0.5:
                if rbm == 0:
                    mult_fact = 0.5
                else:
                    mult_fact = 1.0
            er, ep = self.rbms[rbm].train(visdata, visualize, method=method, ishalf = mult_fact)
            if self.rbms[rbm].double == True:
                wd = 2
            else:
                wd = 1

            visdata = 1.0*(visdata > np.random.random(np.shape(visdata)))
            visdata = self.rbms[rbm].sigmoid(self.rbms[rbm].energy(visdata, wd*self.rbms[rbm].weights, wd*self.rbms[rbm].hid_bias)) 
            #visdata = 1.0*(visdata > np.random.random(np.shape(visdata)))


            errors.append(er); epochs.append(ep)


        return errors, epochs
# ------------------------------------------------------
# Function for one forward pass 
# ------------------------------------------------------
    def forward_pass(self, data, visualize=False, ishalf = 1.0):
        ''' Function for one forward pass to the last layer ''' 
        visdata = data
        visdata = 1.0*(visdata > np.random.random(np.shape(visdata)))
        numrbms = self.numrbms 
        for rbm in range(0, numrbms):
            if ishalf == 0.5:
                if rbm == 0:
                    mult_fact = 0.5
                else:
                    mult_fact = 1.0
            if self.rbms[rbm].double == True:
                wd = 2
            else:
                wd = 1

            visprob = self.rbms[rbm].sigmoid(self.rbms[rbm].energy(visdata, wd*self.rbms[rbm].weights, wd*self.rbms[rbm].hid_bias)) 
            #visdata = 1.0*(visprob > np.random.random(np.shape(visprob)))

        return visdata, visprob 

# *****************************************************************************
    
# ----------------------------------------------------------------------------
# Class for deep neural network
# ----------------------------------------------------------------------------
class Deep_Neural_Network: 
     ''' Class for defining and training deep neural network. 
        At present, the DNN is defined from the DBN'''  
# ----------------------------------------------------------------------------
     def __init__(self, num_layers, num_nodes, dropout = []):
        ''' Function to initialize the DNN '''  

        self.num_layers = num_layers 
        self.numrbms = num_layers - 1
        self.num_nodes = num_nodes   
# -------------------------------------------------------
        self.maxepoch = 100
        self.learning_rate = 0.01
        self.numbatches = 100
        self.initialmomentum = 0.5
        self.finalmomentum = 0.9 
        self.initialepochs = 5 
# -------------------------------------------------------         
# For dropout
# -------------------------------------------------------  
        self.dropout = dropout
        
        dropprob = np.ones(num_layers - 1)
 

        if len(dropout) == 0:                
            dropprob = np.ones(num_layers - 1)
        else:
            dropprob[1:-2] = dropout[1:-2]  

        self.dropprob = dropprob 
        
        

        
# Creating the bottom DBN 
        vis_dbn = Deep_Belief_Network(num_layers-1, num_nodes[0:num_layers-1]) 
# ------------------------------------------------------
# Creating the top RBM   
        top_rbm = Restricted_Boltzmann_Machine(num_nodes[num_layers-1], num_nodes[num_layers-2])
        top_rbm.vistype = 'BB'
        self.vis_dbn = vis_dbn
        self.top_rbm = top_rbm 
        self.create()
# ------------------------------------------------------  

# ------------------------------------------------------
# Class to define layers 
# ------------------------------------------------------
     class Layer: 
        ''' Nested class to define layers of the DNN ''' 
        def __init__(self, bias=[], weights=[], variance = [], top_layer=False):
            ''' Initializing the class layer '''
            self.bias = bias 
            if top_layer == False:
                self.weights = weights
            else: 
                self.variance = variance 



# ------------------------------------------------------
# Function for creating dbm from rbms
# ------------------------------------------------------
     def create(self):   
        ''' Function for creating DNN.
            DNN is defined in terms of layers.  
            Biases are defined for each layer. 
            Weight between layer i and i+1 is 
            defined at layer i'''
# ------------------------------------------------------ 
        vis_dbn = self.vis_dbn
        top_rbm = self.top_rbm  
        num_layers = self.num_layers 
# ------------------------------------------------------        
        dropprob = self.dropprob
# ------------------------------------------------------
        
        layers = []  
        for layer in range(0, num_layers-2):
            print(vis_dbn.num_layers) 
            if layer == 0:
                layers.append(self.Layer(bias=vis_dbn.rbms[layer].vis_bias, weights=vis_dbn.rbms[layer].weights*(1.0/dropprob[layer])))
            else:
                layers.append(self.Layer(bias=0.5*(vis_dbn.rbms[layer].vis_bias+vis_dbn.rbms[layer-1].hid_bias), weights=vis_dbn.rbms[layer].weights*(1.0/dropprob[layer])))
# ------------------------------------------------------ 
# Penultimate layer 
        layers.append(self.Layer(bias=0.5*(top_rbm.hid_bias+vis_dbn.rbms[num_layers-3].hid_bias), weights = (top_rbm.weights.transpose())*(1.0/dropprob[layer])))
# Top layer 
        layers.append(self.Layer(bias=top_rbm.vis_bias, variance=top_rbm.variance, top_layer=True))

        self.layers = layers  
# ------------------------------------------------------
# ------------------------------------------------------
# Options for pretraining 
# ------------------------------------------------------
     def pretrain_options(self, maxepoch, numbatches, learning_rate_dbn, learning_rate_gb, cdn_dbn, 
                         cdn_gb, method = 'Gibbs', initialmomentum=0.5, finalmomentum=0.9, initialepochs=5):
# ------------------------------------------------------
        ''' Specifying the pre-training options '''
        
        self.vis_dbn.options(maxepoch, numbatches, learning_rate_dbn, cdn_dbn, method=method, vistype='BB', initialmomentum=initialmomentum, 
                             finalmomentum=finalmomentum, initialepochs=initialepochs)
        self.top_rbm.options(maxepoch, numbatches, learning_rate_gb, cdn_gb, method=method, vistype='BB', initialmomentum=initialmomentum, 
                             finalmomentum=finalmomentum, initialepochs=initialepochs)

# ------------------------------------------------------
# Pre-training the dbm 
# -----------------------------------------------------
     def pretrain(self, data, target): 
        ''' Pre-training the dnn ''' 
        self.vis_dbn.double = False
        errors_dbn, epochs_dbn = self.vis_dbn.train(data)        
        self.top_rbm.double = False
        #errors_rbm, epochs_rbm = self.top_rbm.train(target)         
# -----------------------------------------------------------
# Updating the DNN  
# -----------------------------------------------------------
        self.create()     

        return errors_dbn, epochs_dbn #, errors_rbm, epochs_rbm
# ------------------------------------------------------
# Options for DNN training 
# ------------------------------------------------------
     def train_options(self, maxepoch, numbatches, learning_rate, initialmomentum=0.5, finalmomentum=0.9, initialepochs=5):
        ''' Specifying the training options ''' 

        self.maxepoch=maxepoch
        self.numbatches = numbatches
        self.learning_rate = learning_rate 
        self.initialmomentum = initialmomentum 
        self.finalmomentum = finalmomentum 
        self.initialepochs = initialepochs
#------------------------------------------------------------------------------
     def train(self, data, target):
         ''' Training the DNN '''
         numbatches = self.numbatches 
         [num_data, num_dim] = np.shape(data) 
        
         num_batch_data = int(floor(num_data/numbatches))
         num_last_batch_data = num_data - (numbatches-1)*num_batch_data 

         perm_data = np.random.permutation(num_data) 

         list_data = []; list_target = []  

         for batch in range(0, numbatches-1):
             indx_batch = perm_data[(batch)*num_batch_data:(batch+1)*num_batch_data]
             list_data.append(data[indx_batch, :]) 
             list_target.append(target[indx_batch, :])

         indx_batch = perm_data[(numbatches-1)*num_batch_data:num_data]
         
         list_data.append(data[indx_batch, :])
         list_target.append(target[indx_batch, :])         
        
         epochs, errors = self.backprop_nadam(list_data, list_target)
         
         for layer in range(0, self.num_layers-1):
             self.layers[layer].weights = (self.layers[layer].weights)*self.dropprob[layer]
             
         
         num_layers = self.num_layers   
         dropprob = np.ones(num_layers - 1)   
         self.dropprob = dropprob
        
         self.epochs = epochs 
         self.errors = errors    
 
         return epochs, errors 
# ------------------------------------------------------
#                Sigmoid activation function 
#           Repeated for the deep neural network
# ------------------------------------------------------
     def sigmoid(self, x): 
         ''' Sigmoid function '''
         return (1.0/(1+np.exp(-x)))

# ------------------------------------------------------
#                     Total energy 
# ------------------------------------------------------
     def energy(self, data, weights, bias):
         ''' Function to calculate energy'''
         [num_data, num_dim] = np.shape(data)
         return np.dot(data, weights) + np.tile(bias,(num_data, 1)) 


# ------------------------------------------------------
#                     Forward pass 
# ------------------------------------------------------
     def predict(self, data): 
         ''' forward pass '''
         probs = self.feedforward(data, self.layers)
         num_layers = self.num_layers
         return probs[num_layers-1]


# ------------------------------------------------------
#                     Forward pass 
# ------------------------------------------------------
     def feedforward(self, data, layers): 
         ''' forward pass '''
         num_layers = self.num_layers
         dropprob = self.dropprob         
         probs = []; #delta = [] 
         probs.append(data)
         for layer in range(1, num_layers):
             bern = 1.0*(dropprob[layer-1]*np.ones(np.shape(data)) > np.random.random(np.shape(data)))
             data = self.sigmoid(self.energy(data*bern, layers[layer-1].weights, layers[layer].bias))
             probs.append(data)
             #delta.append(data*(1.0-data))
         return probs

# ------------------------------------------------------
#            For calculating deltas
# ------------------------------------------------------
     def grads(self, data, target, preds, layers):
         
         num_data = np.shape(data)[0]
         num_layers = self.num_layers
         deltas = [0]*(num_layers-1) # Use better name here
         dw = [0]*(num_layers-1); db = [0]*(num_layers-1)  
         
         err = np.array((preds[num_layers-1] - target)) 
          
         deltas[num_layers-2] = err * preds[num_layers-1] * (1.0 - preds[num_layers-1]) 
         
         for layer in range(num_layers-3, -1, -1):
             deltas[layer] = np.dot(deltas[layer+1], layers[layer+1].weights.T)*preds[layer+1]*(1.0-preds[layer+1])
             
         
         for layer in range(num_layers-2, -1, -1):
             if layer > 0:
                 dw[layer] = -(np.dot(deltas[layer].T, preds[layer])).T#/num_data
             else:
                 dw[layer] = -(np.dot(deltas[layer].T, data)).T#/num_data                       

             db[layer] = -np.sum(deltas[layer], axis = 0)
             
             
             #/num_data
             
         return dw, db
         
# ------------------------------------------------------
#           Backpropagation algorithms
# As a firstcut solution, various SGD methods are imnplemented. 
# Will be replaced by a seperate optimization class   
# ------------------------------------------------------
     def backprop(self, data, target): 
         ''' Backpropagation algorithm using SGD. 
         Momentum method is implemented'''
         numrbms = self.numrbms
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
         maxepoch = self.maxepoch
         numbatches = self.numbatches   
         num_layers = self.num_layers
# Learning rate 
         learning_rate = self.learning_rate
# Momentum    
         initialepochs = self.initialepochs 
         initialmomentum = self.initialmomentum
         finalmomentum = self.finalmomentum
# All the layers are collected here         
         layers = self.layers
# ------------------------------------------------------
#       Initializing the increaments  
#       list of increaments is used for weights and biases
# ------------------------------------------------------ 
         incweights = []; 
         incbiases = []; 
        
         for layer in range(0, num_layers): 
             incbiases.append(np.zeros(np.shape(layers[layer].bias)))
             if layer < num_layers-1:
                 incweights.append(np.zeros(np.shape(layers[layer].weights)))            
# ------------------------------------------------------
         weightcost = 0.001 
         t0 = 300; eta0 = learning_rate        
# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------

         epochs = []; errsums = [];
         for epoch in range(0, maxepoch):
             errsum = 0.0; 
             eta = eta0/(1+(epoch/t0))  
             #eta = learning_rate 
             #eta = 0.1; 
             if eta < 0.0001:
                 eta = 0.0001 
             print('Epoch = ',epoch)        
# Training each batch
             errsum = 0.0   
             for batch in range(0, numbatches): 
                 batchdata = data[batch]
                 num_data = np.shape(batchdata)[0]
                 batchtarget = target[batch]
                 
                 alpha = eta/num_data 
# -----------------------------------------------------------------------------                 
# Prediction using feedforward 
# -----------------------------------------------------------------------------
                 probs = self.feedforward(batchdata,layers)   
                 dw, db = self.grads(batchdata, batchtarget, probs, layers) 
                 
# -----------------------------------------------------------------------------                 
                 predtarget = probs[num_layers-1] 
                 #print(np.max(predtarget), np.max(batchtarget))
                 errsum = errsum + np.sum((batchtarget - predtarget)**2.0)                 
# ------------------------------------------------------
#     Updating weights and biases
# ------------------------------------------------------ 
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 for layer in range(0, num_layers-1): 
                     incbiases[layer+1] = momentum*incbiases[layer+1] + alpha*db[layer]
                     layers[layer+1].bias = layers[layer+1].bias + incbiases[layer+1]
                     incweights[layer] = momentum*incweights[layer] +  alpha*dw[layer] - alpha*num_data*weightcost*incweights[layer] 
                     layers[layer].weights = layers[layer].weights + incweights[layer]


             print('Error = ',errsum)
             epochs.append(epoch); errsums.append(errsum)
# -----------------------------------------------------
# Updated parameters
         self.layers = layers
         return epochs, errsums
         
     
# ------------------------------------------------------
     def backprop_nag(self, data, target): 
         ''' Backpropagation algorithm using SGD. 
         Nesterlov momentum method is implemented'''
         numrbms = self.numrbms
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
         maxepoch = self.maxepoch
         numbatches = self.numbatches   
         num_layers = self.num_layers
# Learning rate 
         learning_rate = self.learning_rate
# Momentum    
         initialepochs = self.initialepochs 
         initialmomentum = self.initialmomentum
         finalmomentum = self.finalmomentum
# All the layers are collected here         
         layers = self.layers
# ------------------------------------------------------
#       Initializing the increaments  
#       list of increaments is used for weights and biases
# ------------------------------------------------------ 
         incweights = []; 
         incbiases = []; 
        
         for layer in range(0, num_layers): 
             incbiases.append(np.zeros(np.shape(layers[layer].bias)))
             if layer < num_layers-1:
                 incweights.append(np.zeros(np.shape(layers[layer].weights)))            
# ------------------------------------------------------
         weightcost = 0.001 
         t0 = 300; eta0 = learning_rate        
# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------

         epochs = []; errsums = [];
         for epoch in range(0, maxepoch):
             errsum = 0.0; 
             eta = eta0/(1+(epoch/t0))  
             #eta = learning_rate 
             #eta = 0.1; 
             if eta < 0.0001:
                 eta = 0.0001 
             print('Epoch = ',epoch)        
# Training each batch
             errsum = 0.0   
             for batch in range(0, numbatches): 
                 batchdata = data[batch]
                 num_data = np.shape(batchdata)[0]
                 batchtarget = target[batch]
                 
                 alpha = eta/num_data 
# -----------------------------------------------------------------------------
# For implementing nsg
# -----------------------------------------------------------------------------
                 nag_layers = layers
# -----------------------------------------------------------------------------
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 for layer in range(0, num_layers-1): 
                     nag_layers[layer+1].bias = layers[layer+1].bias + momentum*incbiases[layer+1]
                     nag_layers[layer].weights = layers[layer].weights + momentum*incweights[layer]

                  
# -----------------------------------------------------------------------------                 
# Prediction using feedforward 
# -----------------------------------------------------------------------------
                 probs = self.feedforward(batchdata,nag_layers)   
                 dw, db = self.grads(batchdata, batchtarget, probs, nag_layers) 
# -----------------------------------------------------------------------------                 
                 predtarget = probs[num_layers-1] 
                 #print(np.max(predtarget), np.max(batchtarget))
                 errsum = errsum + np.sum((batchtarget - predtarget)**2.0)                 
# ------------------------------------------------------
#     Updating weights and biases
# ------------------------------------------------------ 
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 for layer in range(0, num_layers-1): 
                     incbiases[layer+1] = momentum*incbiases[layer+1] + alpha*db[layer]
                     layers[layer+1].bias = layers[layer+1].bias + incbiases[layer+1]
                     incweights[layer] = momentum*incweights[layer] +  alpha*dw[layer] - alpha*num_data*weightcost*incweights[layer] 
                     layers[layer].weights = layers[layer].weights + incweights[layer]


             print('Error = ',errsum)
             epochs.append(epoch); errsums.append(errsum)
# -----------------------------------------------------
# Updated parameters
         self.layers = layers
         return epochs, errsums        
# -----------------------------------------------------        


# ------------------------------------------------------
     def backprop_adagrad(self, data, target): 
         ''' Backpropagation algorithm using SGD. 
         Adagrad method is implemented'''
         numrbms = self.numrbms
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
         maxepoch = self.maxepoch
         numbatches = self.numbatches   
         num_layers = self.num_layers
# Learning rate 
         learning_rate = self.learning_rate
# Momentum    
         initialepochs = self.initialepochs 
         initialmomentum = self.initialmomentum
         finalmomentum = self.finalmomentum
# All the layers are collected here         
         layers = self.layers
         eps = 1e-08 
# ------------------------------------------------------
#       Initializing the increaments  
#       list of increaments is used for weights and biases
# ------------------------------------------------------ 
         incweights = []; 
         incbiases = []; 
         adabiases = [] 
         adaweights = [] 
         for layer in range(0, num_layers): 
             incbiases.append(np.zeros(np.shape(layers[layer].bias)))
             adabiases.append(np.zeros(np.shape(layers[layer].bias)))
             if layer < num_layers-1:
                 incweights.append(np.zeros(np.shape(layers[layer].weights)))            
                 adaweights.append(np.zeros(np.shape(layers[layer].weights)))            
# ------------------------------------------------------
         weightcost = 0.001 
         t0 = 300; eta0 = learning_rate        
# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------

         epochs = []; errsums = [];
         for epoch in range(0, maxepoch):
             errsum = 0.0; 
             #eta = eta0/(1+(epoch/t0))  
             eta = learning_rate 
             #eta = 0.1; 
             if eta < 0.0001:
                 eta = 0.0001 
             print('Epoch = ',epoch)        
# Training each batch
             errsum = 0.0   
             for batch in range(0, numbatches): 
                 batchdata = data[batch]
                 num_data = np.shape(batchdata)[0]
                 batchtarget = target[batch]
                 
                 alpha = eta/num_data                  
# -----------------------------------------------------------------------------                 
# Prediction using feedforward 
# -----------------------------------------------------------------------------
                 probs = self.feedforward(batchdata,layers)   
# Gradients                 
                 dw, db = self.grads(batchdata, batchtarget, probs, layers) 
# -----------------------------------------------------------------------------                 
                 predtarget = probs[num_layers-1] 
                 #print(np.max(predtarget), np.max(batchtarget))
                 errsum = errsum + np.sum((batchtarget - predtarget)**2.0)                                  
# ------------------------------------------------------ 
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 for layer in range(0, num_layers-1): 
# -----------------------------------------------------------------------------                     
#     Sum of square of gradients 
                     adabiases[layer+1] = adabiases[layer+1] + (db[layer]*db[layer])
                     adaweights[layer] = adaweights[layer] + (dw[layer]*dw[layer])
                     
                     
                     incbiases[layer+1] = (eta/np.sqrt(adabiases[layer+1] + eps))*db[layer]
                     layers[layer+1].bias = layers[layer+1].bias + incbiases[layer+1]
                     incweights[layer] = (eta/np.sqrt(adaweights[layer] + eps))*dw[layer] - eta*weightcost*incweights[layer] 
                     layers[layer].weights = layers[layer].weights + incweights[layer]


             print('Error = ',errsum)
             epochs.append(epoch); errsums.append(errsum)
# -----------------------------------------------------
# Updated parameters
         self.layers = layers
         return epochs, errsums        
# -----------------------------------------------------  
# ------------------------------------------------------
     def backprop_adadelta(self, data, target): 
         ''' Backpropagation algorithm using SGD. 
         Adagrad method is implemented'''
         numrbms = self.numrbms
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
         maxepoch = self.maxepoch
         numbatches = self.numbatches   
         num_layers = self.num_layers
# Learning rate 
         learning_rate = self.learning_rate
# Momentum    
         initialepochs = self.initialepochs 
         initialmomentum = self.initialmomentum
         finalmomentum = self.finalmomentum
# All the layers are collected here         
         layers = self.layers
         eps = 1e-08 
# ------------------------------------------------------
#       Initializing the increaments  
#       list of increaments is used for weights and biases
# ------------------------------------------------------ 
         incweights = []; 
         incbiases = []; 
         rmsbiases = [] 
         rmsweights = [] 
         rincbiases = [] 
         rincweights = [] 
         for layer in range(0, num_layers): 
             incbiases.append(np.zeros(np.shape(layers[layer].bias)))
             rmsbiases.append(np.zeros(np.shape(layers[layer].bias)))
             rincbiases.append(np.zeros(np.shape(layers[layer].bias)))
             if layer < num_layers-1:
                 incweights.append(np.zeros(np.shape(layers[layer].weights)))            
                 rmsweights.append(np.zeros(np.shape(layers[layer].weights)))            
                 rincweights.append(np.zeros(np.shape(layers[layer].weights)))            
# ------------------------------------------------------
         weightcost = 0.001 
         t0 = 300; eta0 = learning_rate        
# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------

         epochs = []; errsums = [];
         for epoch in range(0, maxepoch):
             errsum = 0.0; 
             #eta = eta0/(1+(epoch/t0))  
             eta = learning_rate 
             #eta = 0.1; 
             if eta < 0.0001:
                 eta = 0.0001 
             print('Epoch = ',epoch)        
# Training each batch
             errsum = 0.0   
             for batch in range(0, numbatches): 
                 batchdata = data[batch]
                 num_data = np.shape(batchdata)[0]
                 batchtarget = target[batch]
                 
                 alpha = eta/num_data                  
# -----------------------------------------------------------------------------                 
# Prediction using feedforward 
# -----------------------------------------------------------------------------
                 probs = self.feedforward(batchdata,layers)   
# Gradients                 
                 dw, db = self.grads(batchdata, batchtarget, probs, layers) 
# -----------------------------------------------------------------------------                 
                 predtarget = probs[num_layers-1] 
                 #print(np.max(predtarget), np.max(batchtarget))
                 errsum = errsum + np.sum((batchtarget - predtarget)**2.0)                                  
# ------------------------------------------------------ 
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 for layer in range(0, num_layers-1): 
# -----------------------------------------------------------------------------                     
#     Sum of square of gradients 
# -----------------------------------------------------------------------------                     
                     rmsbiases[layer+1] = momentum*rmsbiases[layer+1] + (1.0-momentum)*(db[layer]*db[layer])
                     rmsweights[layer] = momentum*rmsweights[layer] + (1.0-momentum)*(dw[layer]*dw[layer])
                     rincbiases[layer+1] = momentum*rincbiases[layer+1] + (1.0-momentum)*(incbiases[layer+1]*incbiases[layer+1])
                     rincweights[layer] = momentum*rincweights[layer] + (1.0-momentum)*(incweights[layer]*incweights[layer])
                     
                     
                     alpha = (np.sqrt(rincbiases[layer+1] + eps))/(np.sqrt(rmsbiases[layer+1] + eps))
                     
                     incbiases[layer+1] = alpha*db[layer]
                     layers[layer+1].bias = layers[layer+1].bias + incbiases[layer+1]
                     alpha = (np.sqrt(rincweights[layer] + eps))/(np.sqrt(rmsweights[layer] + eps))
                     incweights[layer] = alpha*dw[layer] 
                     layers[layer].weights = layers[layer].weights + incweights[layer] - alpha*weightcost*incweights[layer] 


             print('Error = ',errsum)
             epochs.append(epoch); errsums.append(errsum)
# -----------------------------------------------------
# Updated parameters
         self.layers = layers
         return epochs, errsums        
# -----------------------------------------------------  
# ------------------------------------------------------
     def backprop_adam(self, data, target): 
         ''' Backpropagation algorithm using SGD. 
         Adam method is implemented'''
         numrbms = self.numrbms
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
         maxepoch = self.maxepoch
         numbatches = self.numbatches   
         num_layers = self.num_layers
# Learning rate 
         learning_rate = self.learning_rate
# Momentum    
         initialepochs = self.initialepochs 
         initialmomentum = self.initialmomentum
         finalmomentum = self.finalmomentum
# All the layers are collected here         
         layers = self.layers
         
# ----------------------------------------------------------
# Parameters for adam         
         eps = 1e-08 
         b1 = 0.9
         b2 = 0.999
# ----------------------------------------------------------         
         
# ------------------------------------------------------
#       Initializing the increaments  
#       list of increaments is used for weights and biases
# ------------------------------------------------------ 
         incweights = []; 
         incbiases = []; 
         fmbiases = [] 
         smbiases = [] 
         fmweights = [] 
         smweights = [] 
         for layer in range(0, num_layers): 
             incbiases.append(np.zeros(np.shape(layers[layer].bias)))
             fmbiases.append(np.zeros(np.shape(layers[layer].bias)))
             smbiases.append(np.zeros(np.shape(layers[layer].bias)))
             if layer < num_layers-1:
                 incweights.append(np.zeros(np.shape(layers[layer].weights)))            
                 fmweights.append(np.zeros(np.shape(layers[layer].weights)))            
                 smweights.append(np.zeros(np.shape(layers[layer].weights)))            
# ------------------------------------------------------
         weightcost = 0.001 
         t0 = 300; eta0 = learning_rate        
# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------

         epochs = []; errsums = [];
         for epoch in range(0, maxepoch):
             errsum = 0.0; 
             #eta = eta0/(1+(epoch/t0))  
             eta = learning_rate 
             #eta = 0.1; 
             if eta < 0.0001:
                 eta = 0.0001 
             print('Epoch = ',epoch)        
# Training each batch
             errsum = 0.0   
             t = epoch + 1
             for batch in range(0, numbatches): 
                 batchdata = data[batch]
                 num_data = np.shape(batchdata)[0]
                 batchtarget = target[batch]
                 
                 alpha = eta/num_data                  
# -----------------------------------------------------------------------------                 
# Prediction using feedforward 
# -----------------------------------------------------------------------------
                 probs = self.feedforward(batchdata,layers)   
# Gradients                 
                 dw, db = self.grads(batchdata, batchtarget, probs, layers) 
# -----------------------------------------------------------------------------                 
                 predtarget = probs[num_layers-1] 
                 #print(np.max(predtarget), np.max(batchtarget))
                 errsum = errsum + np.sum((batchtarget - predtarget)**2.0)                                  
# ------------------------------------------------------ 
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 for layer in range(0, num_layers-1): 
# -----------------------------------------------------------------------------                     
#     Sum of square of gradients 
# -----------------------------------------------------------------------------                     
                     fmbiases[layer+1] = b1*fmbiases[layer+1] + (1.0-b1)*(-db[layer])
                     smbiases[layer+1] = b2*smbiases[layer+1] + (1.0-b2)*(db[layer]*db[layer])
                     
                     fmh = fmbiases[layer+1]/(1.0 - b1**t)
                     smh = smbiases[layer+1]/(1.0 - b2**t)
                     
                     incbiases[layer+1] = -alpha/(np.sqrt(smh) + eps)*fmh
                     layers[layer+1].bias = layers[layer+1].bias + incbiases[layer+1]
                     
                     
                     fmweights[layer] = b1*fmweights[layer] + (1.0-b1)*(-dw[layer])
                     smweights[layer] = b2*smweights[layer] + (1.0-b2)*(dw[layer]*dw[layer])
                     
                     fmh = fmweights[layer]/(1.0 - b1**t)
                     smh = smweights[layer]/(1.0 - b2**t)
                     
                     incweights[layer] = -alpha/(np.sqrt(smh) + eps)*fmh
                     
                     layers[layer].weights = layers[layer].weights + incweights[layer] - alpha*weightcost*incweights[layer] 


             print('Error = ',errsum)
             epochs.append(epoch); errsums.append(errsum)
# -----------------------------------------------------
# Updated parameters
         self.layers = layers
         return epochs, errsums        
# -----------------------------------------------------          
# ------------------------------------------------------
     def backprop_adamax(self, data, target): 
         ''' Backpropagation algorithm using SGD. 
         AdaMax method is implemented'''
         numrbms = self.numrbms
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
         maxepoch = self.maxepoch
         numbatches = self.numbatches   
         num_layers = self.num_layers
# Learning rate 
         learning_rate = self.learning_rate
# Momentum    
         initialepochs = self.initialepochs 
         initialmomentum = self.initialmomentum
         finalmomentum = self.finalmomentum
# All the layers are collected here         
         layers = self.layers
         
# ----------------------------------------------------------
# Parameters for adam         
         eps = 1e-08 
         b1 = 0.9
         b2 = 0.999
# ----------------------------------------------------------         
         
# ------------------------------------------------------
#       Initializing the increaments  
#       list of increaments is used for weights and biases
# ------------------------------------------------------ 
         incweights = []; 
         incbiases = []; 
         fmbiases = [] 
         smbiases = [] 
         fmweights = [] 
         smweights = [] 
         for layer in range(0, num_layers): 
             incbiases.append(np.zeros(np.shape(layers[layer].bias)))
             fmbiases.append(np.zeros(np.shape(layers[layer].bias)))
             smbiases.append(np.zeros(np.shape(layers[layer].bias)))
             if layer < num_layers-1:
                 incweights.append(np.zeros(np.shape(layers[layer].weights)))            
                 fmweights.append(np.zeros(np.shape(layers[layer].weights)))            
                 smweights.append(np.zeros(np.shape(layers[layer].weights)))            
# ------------------------------------------------------
         weightcost = 0.001 
         t0 = 300; eta0 = learning_rate        
# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------

         epochs = []; errsums = [];
         for epoch in range(0, maxepoch):
             errsum = 0.0; 
             #eta = eta0/(1+(epoch/t0))  
             eta = learning_rate 
             #eta = 0.1; 
             if eta < 0.0001:
                 eta = 0.0001 
             print('Epoch = ',epoch)        
# Training each batch
             errsum = 0.0   
             t = epoch + 1
             for batch in range(0, numbatches): 
                 batchdata = data[batch]
                 num_data = np.shape(batchdata)[0]
                 batchtarget = target[batch]
                 
                 alpha = eta/num_data                  
# -----------------------------------------------------------------------------                 
# Prediction using feedforward 
# -----------------------------------------------------------------------------
                 probs = self.feedforward(batchdata,layers)   
# Gradients                 
                 dw, db = self.grads(batchdata, batchtarget, probs, layers)
# -----------------------------------------------------------------------------                 
                 predtarget = probs[num_layers-1] 
                 #print(np.max(predtarget), np.max(batchtarget))
                 errsum = errsum + np.sum((batchtarget - predtarget)**2.0)                                  
# ------------------------------------------------------ 
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 for layer in range(0, num_layers-1): 
# -----------------------------------------------------------------------------                     
#     Sum of square of gradients 
# -----------------------------------------------------------------------------                     
                     fmbiases[layer+1] = b1*fmbiases[layer+1] + (1.0-b1)*(-db[layer])
                     smbiases[layer+1] = np.maximum(b2*smbiases[layer+1], np.abs(db[layer]))
                     
                     fmh = fmbiases[layer+1]/(1.0 - b1**t)
                     smh = np.maximum(smbiases[layer+1], eps*np.ones(np.shape(smbiases[layer+1]))) 
# -----------------------------------------------------------------------------                      
                     incbiases[layer+1] = -(alpha/(smh))*fmh
                     layers[layer+1].bias = layers[layer+1].bias + incbiases[layer+1]
                     
                     
                     fmweights[layer] = b1*fmweights[layer] + (1.0-b1)*(-dw[layer])
                     smweights[layer] = np.maximum(b2*smweights[layer], np.abs(dw[layer]))
                     #smweights[layer] = b2*smweights[layer] + (1.0-b2)*(dw[layer]*dw[layer])
                     
                     fmh = fmweights[layer]/(1.0 - b1**t)
                     smh = np.maximum(smweights[layer], eps*np.ones(np.shape(smweights[layer])))  #/(1.0 - b2**t)
                     
                     
                     incweights[layer] = -(alpha/(smh))*fmh
                     
                     layers[layer].weights = layers[layer].weights + incweights[layer] - alpha*weightcost*incweights[layer] 


             print('Error = ',errsum)
             epochs.append(epoch); errsums.append(errsum)
# -----------------------------------------------------
# Updated parameters
         self.layers = layers
         return epochs, errsums        
# -----------------------------------------------------  
# ------------------------------------------------------
     def backprop_nadam(self, data, target): 
         ''' Backpropagation algorithm using SGD. 
         Nadam method is implemented'''
         numrbms = self.numrbms
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
         maxepoch = self.maxepoch
         numbatches = self.numbatches   
         num_layers = self.num_layers
# Learning rate 
         learning_rate = self.learning_rate
# Momentum    
         initialepochs = self.initialepochs 
         initialmomentum = self.initialmomentum
         finalmomentum = self.finalmomentum
# All the layers are collected here         
         layers = self.layers
         
# ----------------------------------------------------------
# Parameters for adam         
         eps = 1e-08 
         b1 = 0.9
         b2 = 0.999
# ----------------------------------------------------------         
         
# ------------------------------------------------------
#       Initializing the increaments  
#       list of increaments is used for weights and biases
# ------------------------------------------------------ 
         incweights = []; 
         incbiases = []; 
         fmbiases = [] 
         smbiases = [] 
         fmweights = [] 
         smweights = [] 
         for layer in range(0, num_layers): 
             incbiases.append(np.zeros(np.shape(layers[layer].bias)))
             fmbiases.append(np.zeros(np.shape(layers[layer].bias)))
             smbiases.append(np.zeros(np.shape(layers[layer].bias)))
             if layer < num_layers-1:
                 incweights.append(np.zeros(np.shape(layers[layer].weights)))            
                 fmweights.append(np.zeros(np.shape(layers[layer].weights)))            
                 smweights.append(np.zeros(np.shape(layers[layer].weights)))            
# ------------------------------------------------------
         weightcost = 0.001 
         t0 = 300; eta0 = learning_rate        
# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------

         epochs = []; errsums = [];
         for epoch in range(0, maxepoch):
             errsum = 0.0; 
             eta = eta0/(1+(epoch/t0))  
             #eta = learning_rate 
             #eta = 0.1; 
             if eta < 0.0001:
                 eta = 0.0001 
             print('Epoch = ',epoch)        
# Training each batch
             errsum = 0.0   
             t = epoch + 1
             for batch in range(0, numbatches): 
                 batchdata = data[batch]
                 num_data = np.shape(batchdata)[0]
                 batchtarget = target[batch]
                 
                 alpha = eta/num_data                  
# -----------------------------------------------------------------------------                 
# Prediction using feedforward 
# -----------------------------------------------------------------------------
                 probs = self.feedforward(batchdata,layers)   
# Gradients                 
                 dw, db = self.grads(batchdata, batchtarget, probs, layers) 
# -----------------------------------------------------------------------------                 
                 predtarget = probs[num_layers-1] 
                 #print(np.max(predtarget), np.max(batchtarget))
                 errsum = errsum + np.sum((batchtarget - predtarget)**2.0)                                  
# ------------------------------------------------------ 
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 for layer in range(0, num_layers-1): 
# -----------------------------------------------------------------------------                     
#     Sum of square of gradients 
# -----------------------------------------------------------------------------                     
                     fmbiases[layer+1] = b1*fmbiases[layer+1] + (1.0-b1)*(-db[layer])
                     smbiases[layer+1] = b2*smbiases[layer+1] + (1.0-b2)*(db[layer]*db[layer])
                     
                     fmh = fmbiases[layer+1]/(1.0 - b1**t)
                     smh = smbiases[layer+1]/(1.0 - b2**t)
                     
                     incbiases[layer+1] = -(alpha/(np.sqrt(smh) + eps))*(b1*fmh + ((1.0-b1)/(1.0 - b1**t))*(-db[layer]))
                     layers[layer+1].bias = layers[layer+1].bias + incbiases[layer+1]
                     
                     
                     fmweights[layer] = b1*fmweights[layer] + (1.0-b1)*(-dw[layer])
                     smweights[layer] = b2*smweights[layer] + (1.0-b2)*(dw[layer]*dw[layer])
                     
                     fmh = fmweights[layer]/(1.0 - b1**t)
                     smh = smweights[layer]/(1.0 - b2**t)
                     
                     incweights[layer] = -(alpha/(np.sqrt(smh) + eps))*(b1*fmh + ((1.0-b1)/(1.0 - b1**t))*(-dw[layer]))
                     
                     layers[layer].weights = layers[layer].weights + incweights[layer] - alpha*weightcost*incweights[layer] 


             print('Error = ',errsum)
             epochs.append(epoch); errsums.append(errsum)
# -----------------------------------------------------
# Updated parameters
         self.layers = layers
         return epochs, errsums        
# -----------------------------------------------------    
        
# ------------------------------------------------------
# ----------- Sigmoid function for general use ---------
# ------------------------------------------------------
def sigmoid(x): 
    ''' Sigmoid function '''
    return (1.0/(1+np.exp(-x)))



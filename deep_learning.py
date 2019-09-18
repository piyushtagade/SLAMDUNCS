import scipy as sc
import numpy as np
from math import floor 
from scipy import linalg
from drawing_tools import *
from process_smiles import * 
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
                                                
             if epoch%10 == 0: 
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

        

        error, epochs = self.contrastive_divergence(list_data, visualize, method=method, ishalf = ishalf)
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
            vis2 = vis2.reshape(1, vis_dim);                                            num_data = np.shape(vis1)[0] 
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
    def rbm_proposal(self, vis, change_indx, weights, vis_bias, hid_bias):
        ''' Proposal distribution based on RBM ''' 
# -----------------------------------------------------------------------------
        if vis.ndim == 1:            
            vis = vis.reshape(1, vis_dim)

        num_data, vis_dim = np.shape(vis)
        vis = 1.0*(vis > np.random.random(np.shape(vis)))
# -----------------------------------------------------------------------------
        idx_change = []; #np.zeros((num_data, num_change*8), dtype=int) 
# -----------------------------------------------------------------------------
# Selecting indices to change
# -----------------------------------------------------------------------------
        for jd in range(0, num_data): 
            change_dim = np.shape(change_indx[jd])[0]    
            num_change = np.random.choice(change_dim, 1, replace=False)[0]
            if num_change == 0:
                num_change =  1
            select_change = np.random.choice(change_indx[jd], num_change, replace=False)    
         
            for i in range(0, num_change):
                for j in range(0, 8):                
                    idx_change.append(select_change[i]*8 + j)         
# -----------------------------------------------------------------------------        
        
# -----------------------------------------------------------------------------
# Obtaining hidden state from present visible state         
        prob_hid = self.sigmoid(self.energy(vis, weights, hid_bias)) 
        hid = 1.0*(prob_hid > np.random.random(np.shape(prob_hid)))
# -----------------------------------------------------------------------------
# Obtaining proposed visisble state from the hidden state                 
        prob_vis = self.sigmoid(self.energy(hid, weights.transpose(), vis_bias))
        vis1 = 1.0*(prob_vis > np.random.random(np.shape(prob_vis)))        
        prop_vis = vis 

        for jd in range(0, num_data):
            idx = idx_change[jd]  

            prop_vis[jd,idx] = vis1[jd,idx]
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
    def proposal(self, vis, fwd_temp, back_temp, change_index = []):
        ''' Function for proposal distribution. 
            Primary purpose is to communicate with the Deep Bayesian Inference
            class ''' 
                

        weights   =  self.weights
        hid_bias  =  self.hid_bias
        vis_bias  =  self.vis_bias
# -----------------------------------------------------------------------------
        prop_vis, prop_ratio = self.rbm_proposal(vis, change_index, 
                                                 fwd_temp*weights, fwd_temp*vis_bias, fwd_temp*hid_bias)
        prior_ratio = self.visible_prob_ratio(prop_vis, vis, weights, 
                                             vis_bias, hid_bias)

        smiles = visualize_smiles(np.array(prop_vis))
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
            visdata = 1.0*(visprob > np.random.random(np.shape(visprob)))

        return visdata, visprob 

# *****************************************************************************
# ----------------------------------------------------------------------------
# Class for deep boltzmann machine 
# ----------------------------------------------------------------------------
class Deep_Boltzmann_Machine: 
    ''' Class for defining and training deep boltzmann machine. 
        At present top layer is defined using Gaussian RBM. 
        The DBM is constructed by concatenating the bottom layer DBN with 
        inverted Gaussian-Bernoulli RBM. '''  
# ----------------------------------------------------------------------------
    def __init__(self, num_layers, num_nodes, connection = None):
        ''' Function to initialize the DBM '''  

        self.num_layers = num_layers 
        self.numrbms = num_layers - 1
        self.num_nodes = num_nodes   
# -------------------------------------------------------
# Default training options for DBM  
        self.cdn = 5 
        self.maxepoch = 100
        self.learning_rate = 0.01
        self.numbatches = 1
        self.initialmomentum = 0.5
        self.finalmomentum = 0.9 
        self.initialepochs = 5 
        self.method = 'Gibbs'
        self.mfiters = 20 
# --------------------------------------------------------
# Checking if the DBM is already trained
        self.is_trained = False 
        self.trained_epochs = 0 
 # -------------------------------------------------------
        if connection == None:
            connect = []
            for layer in range(0, num_layers):
                if layer == 0:
                    connect.append([2])
                if layer == num_layers-1:
                    connect.append([num_layers-1])
                if layer >0 and layer < num_layers-1:
                    connect.append([layer, layer+2])            
            connection = connect
        else:
            connection = connection
            
# -----------------------------------------------------------------------------            
# Storing the connections in sorted order 
        for layer in range(0, num_layers):
            connection[layer] = np.sort(np.array(connection[layer])).tolist() 
        self.connection = connection     
# -----------------------------------------------------------------------------            
# Creating all the RBMs based on the connections  
# -----------------------------------------------------------------------------
        connection = self.connection
        num_vis_rbms = 0
        #connect = connection[0]

        connected_rbms = [] #*num_vis_rbms; 
        # hidden_rbms = [] #*num_hid_rbms;        
        num_rbms = 0
        for layer in range(0, num_layers-1):
            layer_rbms = []
            connect = np.array(connection[layer]) - 1
            for con in range(0, len(connect)):
                if connect[con] > layer and connect[con] != num_layers-1:
                    vis_dim = num_nodes[layer]
                    hid_dim = num_nodes[connect[con]]
                    vis_rbm = Restricted_Boltzmann_Machine(vis_dim, hid_dim)
                    vis_rbm.vis_multfact = len(connect)
                    vis_rbm.hid_multfact = len(connection[connect[con]])
                    layer_rbms.append(vis_rbm) 
                    num_rbms = num_rbms + 1
                if connect[con] == num_layers-1:
                    vis_dim = num_nodes[num_layers-1]
                    hid_dim = num_nodes[layer] 
                    top_rbm = Restricted_Boltzmann_Machine(vis_dim, hid_dim)
                    top_rbm.vistype = 'GB'
                    top_rbm.variance = 1.0*np.ones(vis_dim)
                    top_rbm.hid_multfact = len(connect)
                    top_rbm.vis_multfact = len(connection[connect[con]])

                    layer_rbms.append(top_rbm)
                    num_rbms = num_rbms + 1
                    print("len layer rbms = ", len(layer_rbms))
            connected_rbms.append(layer_rbms)        
                                    
        num_targ_rbms = len(connection[num_layers-1])
# -----------------------------------------------------------------------------        
# -----------------------------------------------------------------------------                                             
        self.num_rbms = num_rbms              
        self.num_targ_rbms = num_targ_rbms              
        
# -----------------------------------------------------------------------------                        
        self.connected_rbms = connected_rbms
 
        self.create()
# ------------------------------------------------------  
# For saving training errors  
# ------------------------------------------------------
        self.epochs = None 
        self.errvis = None 
        self.errtarg = None   
# ------------------------------------------------------
# Class to define layers 
# ------------------------------------------------------
    class Layer: 
        ''' Nested class to define layers of the DBM ''' 
        def __init__(self, connection, bias=[], weights=[], variance = [], top_layer=False):
            ''' Initializing the class layer '''
            self.bias = bias 
            self.connection = connection 
            if top_layer == False:
                self.weights = weights                
            else: 
                self.variance = variance 



# ------------------------------------------------------
# Function for creating dbm from rbms
# ------------------------------------------------------
    def create(self):   
        ''' Function for creating dbm.
            DBM is defined in terms of layers.  
            Biases are defined for each layer. 
            Weight between layer i and i+1 is 
            defined at layer i'''
# ------------------------------------------------------ 
        connected_rbms = self.connected_rbms        
        connection = self.connection         
        num_layers = self.num_layers 
        num_nodes = self.num_nodes
        layers = []
# --------------------------------------------------------        
        for layer in range(0, num_layers-1):
            connect = np.array(connection[layer]) - 1
            weights = []; bias = np.zeros(num_nodes[layer])            
            layers_rbm = connected_rbms[layer]                        
            # For weights
            vrbm = 0
            print(layer, connect, num_layers)
            for con in range(0, len(connect)):
                if connect[con] > layer:
                    if connect[con] !=num_layers - 1:
                        weights.append(layers_rbm[vrbm].weights)
                        bias = bias + layers_rbm[vrbm].vis_bias
                    else:
                        weights.append(layers_rbm[vrbm].weights.T)
                        bias = bias + layers_rbm[vrbm].hid_bias
                    vrbm = vrbm + 1
                else:                    
                    layers_prev_rbm = connected_rbms[connect[con]]
                    rev_connect = np.array(connection[connect[con]]) - 1
                    rev_connect = rev_connect[(rev_connect > connect[con])==True]
                    print("rev connect", rev_connect)                      
                    indx = np.where(rev_connect == layer)[0]
                    print("Else", layer, vrbm, indx, connect[con], len(layers_prev_rbm))                    
                    bias = bias + layers_prev_rbm[indx[0]].hid_bias
            bias = (1.0/len(connect))*bias                                        
            layers.append(self.Layer(connect, bias=bias, weights=weights))    
                    
# For top layer 

        connect = np.array(connection[num_layers-1]) - 1
        bias = np.zeros(num_nodes[num_layers-1]); variance = np.zeros(num_nodes[num_layers-1])
        for con in range(0, len(connect)):
            layer = connect[con]
            layers_rbm = connected_rbms[layer]
            variance = variance + layers_rbm[len(layers_rbm)-1].variance
            bias = bias + layers_rbm[len(layers_rbm)-1].vis_bias                                 
        variance = (1.0/len(connect))*variance                         
        bias = (1.0/len(connect))*bias           
         
        layers.append(self.Layer(connect, bias=bias, variance = variance, top_layer=True))

        self.layers = layers         
# ------------------------------------------------------
# ------------------------------------------------------
# Options for pretraining 
# ------------------------------------------------------
    def pretrain_options(self, maxepoch, numbatches, learning_rate_bb, learning_rate_gb, cdn_bb, 
                         cdn_gb, method = 'Gibbs', initialmomentum=0.5, finalmomentum=0.9, initialepochs=5):
# ------------------------------------------------------
        ''' Specifying the pre-training options '''
        
        num_layers = self.num_layers
        connected_rbms = self.connected_rbms
        for layer in range(0, num_layers-1):
            layer_rbms = connected_rbms[layer]
            num_rbms = len(layer_rbms)
            for rbm in range(0, num_rbms):
                vistype = layer_rbms[rbm].vistype
                if vistype == 'BB':                    
                    layer_rbms[rbm].options(maxepoch, numbatches, learning_rate_bb, cdn_bb, method=method, vistype='BB', initialmomentum=initialmomentum, 
                             finalmomentum=finalmomentum, initialepochs=initialepochs)
                else:
                    layer_rbms[rbm].options(maxepoch, numbatches, learning_rate_gb, cdn_gb, method=method, vistype='GB', initialmomentum=initialmomentum, 
                             finalmomentum=finalmomentum, initialepochs=initialepochs)
            connected_rbms[layer] = layer_rbms
        self.connected_rbms = connected_rbms                  
# ------------------------------------------------------
# Pre-training the dbm 
# -----------------------------------------------------
    def pretrain(self, data, target): 
        ''' Pre-training the dbm ''' 
        
# -----------------------------------------------------------------------------        
        num_layers = self.num_layers
        connection = self.connection
        connected_rbms = self.connected_rbms
        visdata = 1.0*(data > np.random.random(np.shape(data)))          
        print("Visdata = ", np.sum(np.sum(visdata, axis = 0)))
        epochs = []; errors = []
        for layer in range(0, num_layers-1):
# Training the rbms at the visible layer  
            connect = np.array(connection[layer]) - 1
            layer_rbms = connected_rbms[layer]
            num_rbms = len(layer_rbms)
            vrbm = 0
            for con in range(0, len(connect)):
                # Completed till here
                if connect[con] > layer:
                    if connect[con] != num_layers - 1:
                        ep, er = layer_rbms[vrbm].train(visdata)
                    else:
                        print("Target = ", np.sum(np.sum(target, axis = 0)))
                        ep, er = layer_rbms[vrbm].train(target)
                    vrbm = vrbm + 1 
                epochs.append(ep); errors.append(er)               
            if layer + 1 < num_layers - 2:
                print("Inside", layer, num_layers-2)                  
                rev_connect = connect[(connect > layer)==True]
                indx = np.where(rev_connect == layer + 1)[0]
                weights = layer_rbms[indx[0]].weights
                hid_bias = layer_rbms[indx[0]].hid_bias                    
                wd = len(connection[layer+1])
                hid_energy = layer_rbms[indx[0]].energy(visdata, wd*weights, wd*hid_bias)
# Conditional probability of the hidden layer and binary hidden states 
                prob_hid_state = layer_rbms[indx[0]].sigmoid(hid_energy) 
                visdata = 1.0*(prob_hid_state > np.random.random(np.shape(prob_hid_state))) 
                #print("Visdata = ", np.sum(np.sum(visdata, axis = 0)))
                #input("For Pretrain")
                #visdata = hid_states
            connected_rbms[layer] = layer_rbms
        
        self.connected_rbms = connected_rbms                  
        
# -----------------------------------------------------------
# Updating the DBM  
# -----------------------------------------------------------
        self.create() 
#        self.train(visdata, target, pretrain_top=True)
# -----------------------------------------------------------        
        return epochs, errors
# ------------------------------------------------------
# Sigmoid function: used for training the RBM 
# ------------------------------------------------------
    def sigmoid(self, x): 
        ''' Sigmoid function '''
        return (1.0/(1.0+np.exp(-x)))
# ------------------------------------------------------
# Energy of the restricted Boltzmann machine
# ------------------------------------------------------
    def energy(self, data, weights, bias):
        ''' Function to calculate energy for DBM
            Information from all the connected layers is taken into account. '''
            
        num_weights = len(weights)    
        num_data = np.shape(data[0])[0]
        total_energy = np.tile(bias,(num_data, 1))               
        for i in range(0, num_weights):
            w = weights[i]; d = data[i]
            #if np.shape(d)[1] == np.shape(w)[0]:
            total_energy = total_energy + np.dot(d, w)
            #else:
            #    total_energy = total_energy + np.dot(d, w.transpose())        
        return total_energy 

# ---------------------------------------------------------------------------
# Function for sampling target layer from Gauss-Bernoulli DBM 
# ---------------------------------------------------------------------------
    def gb_normal(self, data, weights, bias, variance):
        ''' Sampling from target layer of GB DBM '''
                
        num_weights = len(weights)    
        num_data = np.shape(data[0])[0]
        targ = np.tile(bias,(num_data, 1))               
        for i in range(0, num_weights):
            w = weights[i]; d = data[i]
            targ = targ + np.dot(d, w)
            
        return targ  
# ------------------------------------------------------
# Options for mean field training 
# ------------------------------------------------------
    def train_options(self, maxepoch, numbatches, learning_rate, mfiters=10, cdn=5, method = 'Gibbs', initialmomentum=0.5, finalmomentum=0.9, initialepochs=5):
        ''' Specifying the training options ''' 

        self.maxepoch=maxepoch
        self.numbatches = numbatches
        self.learning_rate = learning_rate 
        self.cdn = cdn
        self.initialmomentum = initialmomentum 
        self.finalmomentum = finalmomentum 
        self.initialepochs = initialepochs
        self.method = method 
        self.mfiters = mfiters   
                
                
                
# -----------------------------------------------------
# Function to factorize the weight matrix
# -----------------------------------------------------
    def factorize(self, weight_fact):
        ''' Function to factorize the weight matrix '''
        
        num_layers = self.num_layers
        
        for layer in range(0, num_layers-1): 
            fact_dim = weight_fact[layer]
            layers = self.layers[layer]
            weight = layers.weights 
            U, s, V = np.linalg.svd(weight, full_matrices=True)
            n = np.shape(U)[0]; p = np.shape(V)[0]        
            min_dim = np.min((n,p))
            if fact_dim > min_dim:
# -------------------------------------------------------
# reduction is not used in factorization            
# -------------------------------------------------------
                fact_dim = min_dim 
            S = np.zeros((fact_dim, p)); Ut = np.zeros((n, fact_dim))
            S[:fact_dim, :fact_dim] = np.diag(s[:fact_dim])
            Ut[:n,:fact_dim] = U[:n,:fact_dim]
            layers.left_fact = Ut
            layers.right_fact = np.dot(S, V) 
            print(np.allclose(np.dot(layers.left_fact, layers.right_fact), np.dot(Ut, np.dot(S,V))))
            self.layers[layer] = layers 
# -----------------------------------------------------
# Function for naive mean field 
# -----------------------------------------------------
    def mean_field(self, visdata, target, layers):
        ''' Mean field approximation for posterior ''' 
#        layers = self.layers
        num_layers = self.num_layers 
        mfiters = self.mfiters
        num_nodes = self.num_nodes   
        num_data = np.shape(visdata)[0]
# --------------------------------------------------- 
# Initialize from random states
# --------------------------------------------------- 
        hidprobs = []#*(num_layers-2) 
        for layer in range(1, num_layers-1):  
            hidprobs.append(np.random.random([num_data, num_nodes[layer]])) 
        for iter in range(0, mfiters):                        
            hidprobs, hid_states = self.hidprobs(hidprobs, visdata, target, layers)                        
        return hidprobs, hid_states            
# ----------------------------------------------------------------------------- 

    def hidprobs(self, hidprobs, visdata, target, layers):
        ''' Function to calculate probability of hidden layers '''
        num_layers = self.num_layers        
        connection = self.connection
        
# For hidden layers         
        for layer in range(1, num_layers-1): 
            connect = np.array(connection[layer]) - 1
            weights = []; data = [] 
            pres_level_connect = connect[(connect > layer) == True]
            bias = layers[layer].bias
            for con in range(0, len(connect)):
                if connect[con] < layer:
                    prev_level_connect = np.array(connection[connect[con]]) - 1
                    prev_level_connect = prev_level_connect[(prev_level_connect > connect[con]) == True]                             
                    indx = np.where(prev_level_connect==layer)[0][0]
                    wt = layers[connect[con]].weights[indx]
                else:
                    indx = np.where(pres_level_connect==connect[con])[0][0]
                    wt = layers[layer].weights[indx].T 
                if connect[con] == 0:
                    dt = visdata
                if connect[con] == num_layers - 1:
                    dt = target
                if connect[con] > 0 and connect[con] < num_layers - 1:
                    dt = hidprobs[connect[con]-1]
                    
                weights.append(wt); data.append(dt)    
            energy = self.energy(data, weights, bias) 
            hidprobs[layer-1] = self.sigmoid(energy)     
                    
        hid_states = []
        for hid_layer in range(0, len(hidprobs)):
            hid_states.append(1.0*(hidprobs[hid_layer] > np.random.random(np.shape(hidprobs[hid_layer]))))            
 
        return hidprobs, hid_states
# -----------------------------------------------------------------------------
# Gibbs sampling for DBM 
# -----------------------------------------------------------------------------
    def gibbs(self, visdata, target, layers, hidprobs = [], burn_period = 0, total_samps = 1, initialize_vis=True, print_steps = False):
        ''' Function for Gibbs sampling from DBM.
            For CD training, supply data and target from training data.
            For sampling from DBM, supply random data and target as input.''' 

# -----------------------------------------------------------------------------
# Getting number of layers and defining hidden layer probabilities
# -----------------------------------------------------------------------------
        num_layers = self.num_layers; num_nodes = self.num_nodes; num_data = np.shape(visdata)[0]
        variance = layers[num_layers-1].variance
        connection = self.connection                 
        if len(hidprobs)==0:
            for layer in range(1, num_layers-1):
                hidprobs.append(np.random.random([num_data, num_nodes[layer]]))
        
        hid = []
        for i in range(0, len(hidprobs)): 
            hid.append(1.0*(hidprobs[i] > np.random.random(np.shape(hidprobs[i]))))
        
        list_vis = []; list_hid = []; list_probs=[]; list_targ = [] 
        visdata = 1.0*(visdata > np.random.random(np.shape(visdata)))        
        for steps in range(0, burn_period+total_samps):
# -----------------------------------------------------------------------------
# For first hidden layer            
# -----------------------------------------------------------------------------
            if print_steps == True:
                print('steps = ', steps)
                
            hidprobs, hid = self.hidprobs(hidprobs, visdata, target/variance, layers)                                 
# -----------------------------------------------------------------------------
# For visible layer            
# -----------------------------------------------------------------------------                            
            connect = np.array(connection[0]) - 1
            
            data = []; weights = []  
            for con in range(0, len(connect)):
                weights.append(layers[0].weights[con].T)
                if connect[con] == num_layers - 1:
                    dt = target/variance
                if connect[con] > 0 and connect[con] < num_layers - 1:
                    dt = hid[connect[con]-1]
                data.append(dt)
                
 
            b = layers[0].bias           
            energy = self.energy(data, weights, b)    
            visprob = self.sigmoid(energy)
            vis = 1.0*(visprob > np.random.random(np.shape(visprob)))
# -----------------------------------------------------------------------------
# For target layer            
# -----------------------------------------------------------------------------                            
            connect = np.array(connection[num_layers-1]) - 1
            weights = []; data = []; bias = layers[num_layers-1].bias 
            variance = layers[num_layers-1].variance          
            for con in range(0, len(connect)):
                wt = layers[connect[con]].weights 
                weights.append(wt[len(wt)-1])
                if connect[con] == 0:
                    dt = vis
                if connect[con] > 0 and connect[con] < num_layers - 1:
                    dt = hid[connect[con]-1]                
                data.append(dt)
                
            targ = self.gb_normal(data, weights, bias, variance) 
            
            if steps >= burn_period:
                list_vis.append(vis)
                list_targ.append(targ)
                list_hid.append(hid)
                list_probs.append(hidprobs)
            visdata = vis; target = targ    
# 
        return list_vis, list_hid, list_probs, list_targ 
# -----------------------------------------------------------------------------
# Training the DBM 
# -----------------------------------------------------------------------------
    def train(self, data, target, method='Gibbs', pretrain_top=False):
        ''' Training the DBM '''
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

        #self.pretrain_top(list_data, list_target)
        errsums, epochs, errsums_vis, errsums_targ = self.contrastive_divergence(list_data, list_target, method=method)
        
        return errsums, epochs, errsums_vis, errsums_targ
# -----------------------------------------------------------------------------
# Function to return weight gradients based on connections 
# -----------------------------------------------------------------------------
    def weight_gradients(self, visdata, target, hiddata): 
        ''' Function for obtaining weight gradients based on connections ''' 
        connection = self.connection
        num_layers = self.num_layers
        incgrad = [] 
        for layer in range(0, num_layers-1): 
            connect = np.array(connection[layer]) - 1
            connect = connect[(connect > layer) == True]                  
            layergrad = []
            if layer == 0:
                dt_pres = visdata 
            else:
                dt_pres = hiddata[layer-1]
                   
            for con in range(0, len(connect)):
                if connect[con] == 0:
                    dt = visdata
                if connect[con] == num_layers - 1:
                    dt = target
                if connect[con] > 0 and connect[con] < num_layers - 1:
                    dt = hiddata[connect[con]-1]
                layergrad.append(np.dot(dt_pres.transpose(), dt)) 
            incgrad.append(layergrad)    
# -----------------------------------------------------------------------------
        return incgrad 
# ------------------------------------------------------

# -----------------------------------------------------------------------------
# Function to return variance gradients based on connections 
# -----------------------------------------------------------------------------
    def variance_gradients(self, visdata, target, hiddata): 
        ''' Function for obtaining weight gradients based on connections ''' 
        connection = self.connection
        num_layers = self.num_layers        
        layers = self.layers
        
        variance = layers[num_layers-1].variance
        num_data = np.shape(target)[0]
        bv = target*variance 
        var1 = 0.5*(bv - np.tile(layers[num_layers-1].bias, (num_data, 1)))**2   
        
# Getting the deviation (second) term 
        targ_connect = np.array(connection[num_layers-1]) - 1
        deviation_sum = np.zeros(np.shape(bv))                       
        for con in range(0, len(targ_connect)):      
            if targ_connect[con] == 0:
                data = visdata
            else:
                data = hiddata[targ_connect[con]-1]

            wt = layers[targ_connect[con]].weights 
            weights = wt[len(wt)-1]                
            deviation_sum = deviation_sum + np.dot(data, weights)                   
        var2 = bv * deviation_sum

                        
        return np.sum((var1 - var2), axis=0)   
# ------------------------------------------------------

# Contrastive divergence algorithm                 
# ------------------------------------------------------
    def contrastive_divergence(self, data, target, method='Gibbs'):
        ''' CD algorithm for training the gb-dbm ''' 
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
        maxepoch = self.maxepoch
        numbatches = self.numbatches   
        num_layers = self.num_layers
# CD(n) algorithm is implemented 
        cdn = self.cdn              
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
        incweights = []; pos_weights = []; neg_weights = [] 
        incbiases = []; pos_bias = []; neg_bias = [] 
        for layer in range(0, num_layers): 
            incbiases.append(np.zeros(np.shape(layers[layer].bias)))
            pos_bias.append(np.zeros(np.shape(layers[layer].bias)))
            neg_bias.append(np.zeros(np.shape(layers[layer].bias))) 
            if layer < num_layers-1:
                layer_incweights = []; layer_posweights = []; layer_negweights = [] 
                
                for wt in range(0, len(layers[layer].weights)):
                    layer_incweights.append(np.zeros(np.shape(layers[layer].weights[wt])))
                    layer_posweights.append(np.zeros(np.shape(layers[layer].weights[wt])))
                    layer_negweights.append(np.zeros(np.shape(layers[layer].weights[wt])))
                
                
                incweights.append(layer_incweights)
                pos_weights.append(layer_posweights)
                neg_weights.append(layer_negweights)
        
        incvariance = np.zeros(np.shape(layers[num_layers-1].variance))

# ------------------------------------------------------
        weightcost = 0.001 
        t0 = 300; eta0 = learning_rate        
# -----------------------------------------------------
# Training the RBM using CD(n) 
# -----------------------------------------------------
        variance = layers[num_layers-1].variance
        epochs = []; errsums = [];  errsums_vis = []; errsums_targ = []
        connection = self.connection 
# ----------------------------------------------------
        start_epoch = self.trained_epochs 
# ----------------------------------------------------
        for epoch in range(start_epoch, maxepoch):
             errsum = 0.0; errvis = 0.0; errtarg = 0.0
             eta = eta0/(1+(epoch/t0))  
             #eta = learning_rate 
             if eta < 0.00001:
                 eta = 0.00001 
             #print('Epoch = ',epoch)        
# Training each batch
             for batch in range(0, numbatches): 
                 variance = layers[num_layers-1].variance                                  
                 zvar = np.log(variance)                                                                         
# Retrive batchdata from the data              
                 batchdata = data[batch]
                 [num_data, num_dim] = np.shape(batchdata)
# Retrive batchtarget from target                  
                 batchtarget = target[batch]
# Learning rate                  
                 alpha = eta/num_data   
# -----------------------------------------------------------------------------
# Start of positive phase 
# -----------------------------------------------------------------------------                   
                 batchdata = 1.0*(batchdata > np.random.random(np.shape(batchdata)))                  
                 batchtarget = batchtarget/variance
                 
# -----------------------------------------------------------------------------
# Mean field
# -----------------------------------------------------------------------------                   
                 hidprobs, hid_states = self.mean_field(batchdata, batchtarget, layers)
                 
# Weight gradients 
                 pos_weights = self.weight_gradients(batchdata, batchtarget, hidprobs)
                 
                 
# Biases                 
# For visible layer
                 pos_bias[0] = np.sum(batchdata, axis=0)                                                                                     
# For middle hidden layers                  
                 for layer in range(1, num_layers-2): 
                     pos_bias[layer] = np.sum(hidprobs[layer-1], axis=0) 
# -----------------------------------------------------------------------------                     
# For penultimate hidden layer
                 layer = num_layers - 2
                 pos_bias[layer] = np.sum(hidprobs[layer-1], axis=0) 
# For target layer                 
                 layer = num_layers - 1
                 pos_bias[layer] = np.sum(batchtarget, axis=0)                        
                 
                 pos_var = self.variance_gradients(batchdata, batchtarget, hidprobs) 
                 
# -----------------------------------------------------------------------------
# End of positive phase 
# -----------------------------------------------------------------------------                                            

# -----------------------------------------------------------------------------
# CDn sampling. At present only Gibbs sampling is kept 
# ----------------------------------------------------------------------------- 
                 list_vis, list_hid, list_probs, list_targ = self.gibbs(batchdata, batchtarget*variance, layers, hidprobs = hidprobs, total_samps = cdn, initialize_vis = True)
                 negdata = list_vis[cdn-1]; hid_states = list_hid[cdn-1]; negtarget = list_targ[cdn-1]; hidprobs = list_probs[cdn-1]
                 negtarget = negtarget/variance
                 errsum = errsum + np.sum((batchdata - negdata)**2.0) + np.sum((batchtarget*variance - negtarget*variance)**2.0)
                 errvis = errvis + np.sum((batchdata - negdata)**2.0)
                 errtarg = errtarg + np.sum((batchtarget*variance - negtarget*variance)**2.0)
                 #print(np.min(negtarget, axis=0), np.max(negtarget, axis=0), np.min(batchtarget, axis=0), np.max(batchtarget, axis=0))
# -----------------------------------------------------------------------------
# Start of negative phase
# -----------------------------------------------------------------------------                         
# Weight gradients 
                 neg_weights = self.weight_gradients(negdata, negtarget, hidprobs)


# Biases                 
                 neg_bias[0] = np.sum(negdata, axis=0) 
# For middle hidden layers                  
                 for layer in range(1, num_layers-2): 
                     neg_bias[layer] = np.sum(hidprobs[layer-1], axis=0) 
# -----------------------------------------------------------------------------                     
# For penultimate hidden layer
                 layer = num_layers - 2
                 neg_bias[layer] = np.sum(hidprobs[layer-1], axis=0) 
# For target layer                 
#                 variance = layers[num_layers-1].variance
                 layer = num_layers - 1
                 neg_bias[layer] = np.sum(negtarget, axis=0) 


                 neg_var = self.variance_gradients(negdata, negtarget, hidprobs)
# -----------------------------------------------------------------------------
# End of negative phase 
# -----------------------------------------------------------------------------
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 for layer in range(0, num_layers): 
                     if layer == num_layers-1:
                         alp = alpha/10
                     else:
                         alp = alpha
                             
                     incbiases[layer] = momentum*incbiases[layer] + alp*(pos_bias[layer]-neg_bias[layer])                     
                     layers[layer].bias = layers[layer].bias + incbiases[layer]
                     if layer < num_layers-1:
                         if layer == num_layers-2:
                             alp = alpha/10
                         else:
                             alp = alpha
                         num_weights = len(layers[layer].weights) 
                         
                         for con in range(0, num_weights):                         
                             incweights[layer][con] = momentum*incweights[layer][con] +  alp*(pos_weights[layer][con] - neg_weights[layer][con]) - alp*num_data*weightcost*incweights[layer][con] 
                             layers[layer].weights[con] = layers[layer].weights[con] + incweights[layer][con]  
                     if epoch > 100:
                         incvariance = momentum*incvariance + 1e-04*np.exp(-zvar)*alpha*(pos_var - neg_var)                          
                         zvar = zvar + incvariance
#zvar = zvar + incvariance
                         variance = np.exp(zvar)
                         variance[np.where(variance < 1e-05)] = 1e-05
                         zvar = np.log(variance) 
                         layers[num_layers-1].variance = variance
             
                
                
             if epoch%10 == 0: 
                 print('Epoch = ', epoch, 'Error = ',errsum, errvis, errtarg, 'Learning rate = ', alpha)   
             #print('Error = ',errsum, errvis, errtarg)
             epochs.append(epoch); errsums.append(errsum); errsums_vis.append(errvis); errsums_targ.append(errtarg)
# -----------------------------------------------------
# Updated parameters 
        self.layers = layers
        self.epochs = epochs 
        self.errvis = errsums_vis 
        self.errtarg = errsums_targ 
        self.is_trained = True
        self.trained_epochs = maxepoch  
   

        return errsums, epochs, errsums_vis, errsums_targ 

# -----------------------------------------------------------
# Pre-training the penultimate layer 
# -----------------------------------------------------------
    def pretrain_top(self, pretrain_data, pretrain_target):
        ''' CD algorithm for pre-training the top layer of DBM ''' 
# ------------------------------------------------------
#       initializing the relevant variables 
# ------------------------------------------------------
        connection = self.connection
        num_layers = self.num_layers
        targ_connect = np.array(connection[num_layers-1]) - 1
        targ_connect = targ_connect[(targ_connect > 0) == True]
        print(targ_connect)
# Number of top DBMs to be pretrained.
        num_top_dbms = len(targ_connect)
# Collecting the visdata                                 
        visdata = 1.0*(pretrain_data > np.random.random(np.shape(pretrain_data)))
        print("Visdata = ", np.sum(np.sum(visdata, axis = 0)))
        collection_visdata = [visdata]
        for layer in range(0, num_layers-2): 
            connect = np.array(connection[layer]) - 1
            layer_rbms = self.connected_rbms[layer]                  
            if layer + 1 < num_layers - 2:
                print("Inside", layer, num_layers-2)                  
                rev_connect = connect[(connect > layer)==True]
                indx = np.where(rev_connect == layer + 1)[0]
                weights = layer_rbms[indx[0]].weights
                hid_bias = layer_rbms[indx[0]].hid_bias                    
                wd = len(connection[layer+1])
                print("Layers wd", layer+1, wd)
                hid_energy = layer_rbms[indx[0]].energy(visdata, wd*weights, wd*hid_bias)
# Conditional probability of the hidden layer and binary hidden states 
                visdata = layer_rbms[indx[0]].sigmoid(hid_energy)                                              
                collection_visdata.append(visdata)
                visdata = 1.0*(visdata > np.random.random(np.shape(visdata)))
                print("Pretrain top visdata = ", np.sum(np.sum(visdata, axis = 0)))
        
        print(len(collection_visdata))
        
        for con in range(0, num_top_dbms):
            hid_con = targ_connect[con]
            hid_connect = np.array(connection[hid_con]) - 1
            vis_con = np.max(hid_connect[(hid_connect < hid_con) == True])
            dbm_connect = [vis_con, hid_con, num_layers - 1] 
            print(vis_con, hid_con, dbm_connect)
            visdata = collection_visdata[vis_con]
            print("Going Inside Visdata = ", np.sum(np.sum(visdata, axis = 0)))
            epochs, errsums, errtargs = self.pretrain_dbm(visdata, pretrain_target, dbm_connect)
         
# -----------------------------------------------------------
# Pre-training the penultimate layer 
# -----------------------------------------------------------
    def pretrain_dbm(self, visdata, pretrain_target, dbm_connect):
        ''' CD algorithm for pre-training the top layer of DBM ''' 

# ------------------------------------------------------
#       Creating batches 
# ------------------------------------------------------
        #visdata = visprob   
        numbatches = self.numbatches 
        [num_data, num_dim] = np.shape(visdata) 
        
        num_batch_data = int(floor(num_data/numbatches))
        
        perm_data = np.random.permutation(num_data) 

        data = []; target = []  
        print("dbm_connect = ", dbm_connect)  
        for batch in range(0, numbatches-1):
            indx_batch = perm_data[(batch)*num_batch_data:(batch+1)*num_batch_data]
            data.append(visdata[indx_batch, :]) 
            target.append(pretrain_target[indx_batch, :])

        indx_batch = perm_data[(numbatches-1)*num_batch_data:num_data]
         
        data.append(visdata[indx_batch, :])
        target.append(pretrain_target[indx_batch, :])

# ------------------------------------------------------
#       Three layers are pretrained 
# ------------------------------------------------------
        maxepoch = self.maxepoch
        numbatches = self.numbatches   
        layers = self.layers
        num_layers = self.num_layers
# ------------------------------------------------------
#       Connection 
# ------------------------------------------------------
        connection = self.connection        
# ------------------------------------------------------
#       Top three layers are pretrained 
# ------------------------------------------------------
        top_layers = 3 
        vis_con = dbm_connect[0]
        hid_con = dbm_connect[1]
        targ_con = dbm_connect[2]
        
        vis_bias = layers[vis_con].bias
        hid_bias = layers[hid_con].bias
        lab_bias = layers[targ_con].bias
        print(np.shape(lab_bias))                 
        
        print("no_weights = ", len(layers[dbm_connect[0]].weights))                  
        vis_connect = np.array(connection[vis_con]) - 1 
        vis_connect = vis_connect[(vis_connect > vis_con) == True]                      
        hid_connect = np.array(connection[hid_con]) - 1         
        hid_connect = hid_connect[(hid_connect > hid_con) == True]
        targ_connect = np.array(connection[targ_con]) - 1 
        
# -----------------------------------------------------------------------------                               
        vis_multfact = len(connection[vis_con])
        hid_multfact = len(connection[hid_con])
        targ_multfact = len(connection[targ_con])                         
                               
        print('vis_conn = ', vis_connect, hid_connect)
        
        vishid_con = np.where(vis_connect == hid_con)[0][0]
        hidlab_con = np.where(hid_connect == targ_con)[0][0]
        print('vishid_con = ', vishid_con, hidlab_con)
        print('vis_con=', vis_con, hid_con, targ_con)
        vishid = layers[vis_con].weights[vishid_con]
        hidlab = layers[hid_con].weights[hidlab_con]
                       
        variance = layers[targ_con].variance               
# ------------------------------------------------------
#       Initializing the increaments  
# ------------------------------------------------------ 
        incvishid = np.zeros(np.shape(vishid))
        inchidlab = np.zeros(np.shape(hidlab))
        
        
        
        incvisbias = np.zeros(np.shape(vis_bias))
        inchidbias = np.zeros(np.shape(hid_bias))
        inclabbias = np.zeros(np.shape(lab_bias))
        incvariance = np.zeros(np.shape(variance))
# CD(n) algorithm is implemented 
        cdn = 5 # This is fixed at the moment              
# Learning rate 
        learning_rate = self.learning_rate # This is fixed at the moment              
# Momentum    
        initialepochs = self.initialepochs 
        initialmomentum = self.initialmomentum
        finalmomentum = self.finalmomentum 
        weightcost = 0.001 
        t0 = 300; eta0 = learning_rate        
# ------------------------------------------------------------
# Pre-training top three layers of DBM using CD(n) algorithm
# ------------------------------------------------------------
        epochs = []; errsums = []; errtargs = [] 
        for epoch in range(0, maxepoch):
             errsum = 0.0; errtarg = 0.0 
             eta = eta0/(1+(epoch/t0))  
             #eta = learning_rate 
             if eta < 0.0001:
                 eta = 0.0001 
             #print('Epoch = ',epoch)        
# Training each batch
             for batch in range(0, numbatches):                 
# Retrive batchdata from the data              
                 batchdata = data[batch]
                 batchtarget = target[batch]
                 [num_data, num_dim] = np.shape(batchdata)
                 alpha = eta/num_data  
# Convert the data to binary 
                 batchdata = 1.0*(batchdata > np.random.random(np.shape(batchdata))) 
                 batchtarget = batchtarget/variance    
                 zvar = np.log(variance)
# Total energy to the hidden layer
                 wd = hid_multfact - 1
                 d1 = batchdata; d2 = batchtarget
                 w1 = vishid; w2 = hidlab 
                 b = hid_bias
                 #print("Before hid_energy", (np.sum(np.sum(batchdata, axis = 0))), (np.sum(np.sum(batchtarget, axis = 0))))
                 hid_energy = self.energy([d1, d2], [wd*w1, wd*w2.T], wd*b)                 
                 #print("After hid_energy", (np.sum(np.sum(hid_energy, axis = 0))))
                 #input("Hid Energy")
                 
# Conditional probability of the hidden layer and binary hidden states 
                 prob_hid_state = self.sigmoid(hid_energy) 
                 hid_states = 1.0*(prob_hid_state > np.random.random(np.shape(prob_hid_state))) 
                 
# Data dependent 
                 pos_vishid = np.dot(batchdata.transpose(),prob_hid_state)
                 pos_hidlab = np.dot(prob_hid_state.transpose(),batchtarget)  
                 

                 pos_visbias = np.sum(batchdata, axis=0)
                 pos_labbias = np.sum(batchtarget, axis=0)
                 pos_hidbias = np.sum(prob_hid_state, axis=0)
# For variance                 
                 bv = batchtarget*variance 
                 var1 = 0.5*(bv - np.tile(lab_bias, (num_data, 1)))**2   
                 var2 = bv * (np.dot(hidlab.T, prob_hid_state.T).T)
                 pos_var = np.sum((var1 - var2), axis=0)                   
# Starting the negative phase
                 for cditer in range(0, cdn):
                     wd = targ_multfact
                     negtarget = self.gb_normal([hid_states], [wd*hidlab], wd*lab_bias, variance)/variance  
                     
# Total energy to the visible layer
                     wd = vis_multfact
                     d1 = hid_states; d2 = []
                     w1 = wd*vishid.T; w2 = [] 
                     b = wd*vis_bias
                     vis_energy = self.energy([d1], [w1], b) 
                     negdata = self.sigmoid(vis_energy)
                     negdata = 1.0*(negdata > np.random.random(np.shape(negdata)))
                     
# Total energy to the hidden layer
                     wd = hid_multfact - 1 
                     d1 = negdata; d2 = negtarget
                     w1 = vishid; w2 = hidlab 
                     b = hid_bias
                     hid_energy = self.energy([d1, d2], [wd*w1, wd*w2.T], wd*b)                
# Conditional probability of the hidden layer and binary hidden states 
                     prob_hid_state = self.sigmoid(hid_energy) 
                     hid_states = 1.0*(prob_hid_state > np.random.random(np.shape(prob_hid_state))) 
                     
# -----------------------------------------------------------------------------
# Reconstruction error
                 errsum = errsum + np.sum((batchdata - negdata)**2.0) #/(num_data*num_dim)
                 #errtarg = errtarg + np.sum((batchtarget*variance - negtarget*variance)**2.0)
                 errtarg = errtarg + np.sum((batchtarget - negtarget)**2.0)
# -----------------------------------------------------------------------------
# Model dependent tersm                 
                 neg_vishid = np.dot(negdata.transpose(),prob_hid_state)
                 neg_hidlab = np.dot(prob_hid_state.transpose(), negtarget) 

                 
                 neg_visbias = np.sum(negdata, axis=0)
                 neg_labbias = np.sum(negtarget, axis=0)
                 neg_hidbias = np.sum(prob_hid_state, axis=0)
# For variance                 
                 bv = negtarget*variance 
                 var1 = 0.5*(bv - np.tile(lab_bias, (num_data, 1)))**2   
                 var2 = bv * (np.dot(hidlab.T, prob_hid_state.T).T)
                 neg_var = np.sum((var1 - var2), axis=0)                                                     
# Setting up the momentum
                 if epoch < initialepochs:
                     momentum = initialmomentum 
                 else:
                     momentum = finalmomentum  
# Obtaining the increament 
                 incvishid = momentum*incvishid + alpha*(pos_vishid - neg_vishid) - alpha*num_data*weightcost*incvishid 
                 inchidlab = momentum*inchidlab + 0.1*alpha*(pos_hidlab - neg_hidlab) - 0.1*alpha*num_data*weightcost*inchidlab

                 
                 incvisbias = momentum*incvisbias + alpha*(pos_visbias - neg_visbias)
                 inchidbias = momentum*inchidbias + alpha*(pos_hidbias - neg_hidbias) 
                 inclabbias = momentum*inclabbias + 0.1*alpha*(pos_labbias - neg_labbias)      
                 
                 
                 
                     
# Updating the parameters 

                 

                 vishid = vishid + incvishid 
                 hidlab = hidlab + inchidlab
                 
                 
                 vis_bias = vis_bias +  incvisbias
                 hid_bias = hid_bias +  inchidbias
                 lab_bias = lab_bias + inclabbias
                 
                 
                 
                 if epoch > 100:
                     incvariance = momentum*incvariance + 1e-05*np.exp(-zvar)*alpha*(pos_var - neg_var)  
                     zvar = zvar + incvariance
#zvar = zvar + incvariance
                     variance = np.exp(zvar)
                     variance[np.where(variance < 1e-05)] = 1e-05
                     zvar = np.log(variance)                     
                                                
             if epoch%10 == 0: 
                 print('Epoch = ', epoch, 'Error = ',errsum, errtarg) 
             #print('Error = ',errsum, errtarg)
             epochs.append(epoch); errsums.append(errsum); errtargs.append(errtarg)
# -----------------------------------------------------
# Updated parameters


         
        layers[vis_con].bias = vis_bias
        layers[hid_con].bias = hid_bias 
        layers[targ_con].bias = lab_bias

                        
        layers[vis_con].weights[vishid_con] = vishid
        layers[hid_con].weights[hidlab_con] = hidlab 

        layers[targ_con].variance = variance 

        self.layers = layers

        return epochs, errsums, errtargs
# -----------------------------------------------------



# ------------------------------------------------------
# Function to sample from dbm 
# ------------------------------------------------------
    def sample(self, no_samples, burn_period, no_steps):
        ''' Function for sampling from DBM using alternating Gibbs sampling '''
# Number of nodes in each layer        
        num_nodes = self.num_nodes
        num_layers = self.num_layers
        layers = self.layers
        
        # For visible layer
        data = np.random.random([no_samples, num_nodes[0]])
        data = 1.0*(data > np.random.random([no_samples, num_nodes[0]]))
        
        # For hidden layers
        hidprobs = []
        for layer in range(1, num_layers-1): 
            hidprobs.append(np.random.random([no_samples, num_nodes[layer]]))
        
        # For target layer
        target = np.random.random([no_samples, num_nodes[num_layers-1]])
        list_vis, list_hid, list_probs, list_targ = self.gibbs(data, target, layers, hidprobs = hidprobs, burn_period = burn_period, 
                                                               total_samps = no_steps, initialize_vis = True, print_steps = True)
        
        return list_vis, list_targ     

# -----------------------------------------------------------------------------
# For prediction using Monte Carlo sampling
# ----------------------------------------------------------------------------- 
    def predict_mc(self, vis, burn_period=5000, no_samples=1000):
        ''' Function for prediction using MC '''
        num_nodes = self.num_nodes
        num_layers = self.num_layers
        layers = self.layers
        num_data = np.shape(vis)[0]
        # For visible layer        
        # For hidden layers
        hidprobs = []
        for layer in range(1, num_layers-1): 
            hidprobs.append(np.random.random([num_data, num_nodes[layer]]))
        
        # For target layer
        targ = np.random.random([num_data, num_nodes[num_layers-1]]); target_samps = []; target = np.zeros([num_data, num_nodes[num_layers-1]]) 
        for step in range(0, burn_period+no_samples):
            print('Step = ',step)
            data = vis
            list_vis, list_hid, list_probs, list_targ = self.gibbs(data, targ, layers, hidprobs = hidprobs, burn_period = 0, 
                                                               total_samps = 1, initialize_vis = True)      
            hidprobs = list_probs[0]; targ = list_targ[0]
            if step >=burn_period:
                target_samps.append(targ)
                target = target + np.array(targ)
        

        
        return target/no_samples, target_samps        

# -----------------------------------------------------------------------------
# Defined primarily to communicate with the Deep Bayesian Inference class
# ----------------------------------------------------------------------------- 
    def predict(self, vis): 
        ''' Function for communicating with Deep Bayesian Inference class '''        
        pred_mean, pred_var, samps = self.predict_mcg(vis, burn_period=8000, no_samples=2000)
        
        num_samps = len(samps)
        if pred_mean.ndim == 1:
            num_dim = np.shape(pred_mean)
            num_data = 1
        else:
            num_data, num_dim = np.shape(pred_mean)
        

        pred_sd = np.sqrt(pred_var - np.square(pred_mean)) 
        #pred_mean = pred_mean[0,:]        
        return pred_mean[:,2:4], pred_sd[:,2:4] 
    
    
# -----------------------------------------------------------------------------
# Prediction using alternate Gibbs sampling
# -----------------------------------------------------------------------------
    def predict_mcg(self, vis, burn_period=10000, no_samples=5000):
        ''' Function for prediction using alternate Gibbs sampling MC '''
        num_nodes = self.num_nodes
        num_layers = self.num_layers
        layers = self.layers
        num_data = np.shape(vis)[0]
        variance = layers[num_layers-1].variance

        hidprobs = []
        for layer in range(1, num_layers-1): 
            hidprobs.append(np.random.random([num_data, num_nodes[layer]]))
        
        hid = []
        for i in range(0, len(hidprobs)): 
            hid.append(1.0*(hidprobs[i] > np.random.random(np.shape(hidprobs[i]))))
# For target layer
        targ = np.random.random([num_data, num_nodes[num_layers-1]]); target_samps = []; target = np.zeros([num_data, num_nodes[num_layers-1]])
        target_var = np.zeros([num_data, num_nodes[num_layers-1]])
# -----------------------------------------------------------------------------
# Getting number of layers and defining hidden layer probabilities
# -----------------------------------------------------------------------------                
        list_vis = []; list_hid = []; list_probs=[]; list_targ = []         
        for steps in range(0, burn_period+no_samples):
# -----------------------------------------------------------------------------
# For first hidden layer            
# -----------------------------------------------------------------------------
            #if steps%100 == 0:  
            #    print("Steps = ", steps) 
            vis = 1.0*(vis > np.random.random(np.shape(vis))) 
            data = vis 
            list_vis, list_hid, list_probs, list_targ = self.gibbs(data, targ, layers, hidprobs = hidprobs, burn_period = 0, 
                                                               total_samps = 1, initialize_vis = True)      
            hidprobs = list_probs[0]; targ = list_targ[0]            
            
            target_samps.append(targ)                
            
            if steps >= burn_period:
                target = target + np.array(targ)
                target_var = target_var + np.square(np.array(targ))            
# 
        return target/no_samples, target_var/no_samples, target_samps 
# -----------------------------------------------------------------------------    
    
# ----------------------------------------------------------------------------
# Class for deep neural network
# ----------------------------------------------------------------------------
class Deep_Neural_Network: 
     ''' Class for defining and training deep neural network. 
        At present, the DNN is defined from the DBN'''  
# ----------------------------------------------------------------------------
     def __init__(self, num_layers, num_nodes):
        ''' Function to initialize the DBM '''  

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
# Creating the bottom DBN 
        vis_dbn = Deep_Belief_Network(num_layers-1, num_nodes[0:num_layers-1]) 
# ------------------------------------------------------
# Creating the top RBM   
        top_rbm = Restricted_Boltzmann_Machine(num_nodes[num_layers-1], num_nodes[num_layers-2])
        top_rbm.vistype = 'GB'
        self.vis_dbn = vis_dbn
        self.top_rbm = top_rbm 
        self.create()
# ------------------------------------------------------  

# ------------------------------------------------------
# Class to define layers 
# ------------------------------------------------------
     class Layer: 
        ''' Nested class to define layers of the DBM ''' 
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
        ''' Function for creating dbm.
            DBM is defined in terms of layers.  
            Biases are defined for each layer. 
            Weight between layer i and i+1 is 
            defined at layer i'''
# ------------------------------------------------------ 
        vis_dbn = self.vis_dbn
        top_rbm = self.top_rbm  
        num_layers = self.num_layers 
        layers = []  
        for layer in range(0, num_layers-2):
            print(vis_dbn.num_layers) 
            if layer == 0:
                layers.append(self.Layer(bias=vis_dbn.rbms[layer].vis_bias, weights=vis_dbn.rbms[layer].weights))
            else:
                layers.append(self.Layer(bias=0.5*(vis_dbn.rbms[layer].vis_bias+vis_dbn.rbms[layer-1].hid_bias), weights=vis_dbn.rbms[layer].weights))
# ------------------------------------------------------ 
# Penultimate layer 
        layers.append(self.Layer(bias=0.5*(top_rbm.hid_bias+vis_dbn.rbms[num_layers-3].hid_bias), weights = top_rbm.weights.transpose()))
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
        self.top_rbm.options(maxepoch, numbatches, learning_rate_gb, cdn_gb, method=method, vistype='GB', initialmomentum=initialmomentum, 
                             finalmomentum=finalmomentum, initialepochs=initialepochs)

# ------------------------------------------------------
# Pre-training the dbm 
# -----------------------------------------------------
     def pretrain(self, data, target): 
        ''' Pre-training the dnn ''' 
        self.vis_dbn.double = False
        errors_dbn, epochs_dbn = self.vis_dbn.train(data)        
        self.top_rbm.double = True
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
        
         epochs, errors = self.backprop(list_data, list_target)
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
         probs = []; #delta = [] 
         probs.append(data)
         for layer in range(1, num_layers):
             data = self.sigmoid(self.energy(data, layers[layer-1].weights, layers[layer].bias))
             probs.append(data)
             #delta.append(data*(1.0-data))
         return probs

# ------------------------------------------------------
#            For calculating deltas
# ------------------------------------------------------
     def deltas(self, data, target, probs, layers):
         
         num_data = np.shape(data)[0]
         num_layers = self.num_layers
         Ix = [0]*num_layers # Use better name here
         dw = [0]*(num_layers-1); db = [0]*(num_layers-1)  
         Ix[num_layers-1] = np.array((probs[num_layers-1] - target)) 
         
         for layer in range(num_layers-2, 0, -1):             
             Ix[layer] = np.dot(Ix[layer+1], layers[layer].weights.T)*probs[layer]*(1.0-probs[layer])
             db[layer] = -np.sum(Ix[layer+1], axis = 0)/num_data
             if layer > 0:
                 dw[layer] = -(np.dot(Ix[layer+1].T, probs[layer])).T/num_data
             else:
                 dw[layer] = -(np.dot(Ix[layer+1].T, data)).T/num_data          
         return dw, db
         
# ------------------------------------------------------
#               Backpropagation algorithm  
# ------------------------------------------------------
     def backprop(self, data, target): 
         ''' Backpropagation algorithm '''
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
             eta = learning_rate 
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
                 dw, db = self.deltas(batchdata, batchtarget, probs, layers) 
                 
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
# ----------- Sigmoid function for general use ---------
# ------------------------------------------------------
def sigmoid(x): 
    ''' Sigmoid function '''
    return (1.0/(1+np.exp(-x)))



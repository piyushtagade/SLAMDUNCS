# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:01:03 2017

@author: piyush.t
"""

# -------------------------------------------------------
# Design of experiment methodology for binary dataset
# Key idea is to select training dataset such that distance 
# between points is maximized.  
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


# ----------------------------------------------------------------------
# Function to define similarity measure between two binary vectors
# ----------------------------------------------------------------------
def distance(bin1, bin2):
# ----------------------------------------------------------------------    
    a = np.sum(bin1*bin2); b = np.sum((1.0-bin1)*bin2); 
    c = np.sum(bin1*(1.0-bin2)); d = np.sum((1.0-bin1)*(1.0-bin2)) 
# --------------------------------------------------------------------------
# Initially implementing YULEQ/Hamming distance. Other distances and similarity 
# measures will be explored in the future. 
# --------------------------------------------------------------------------
    dist = (2.0*b*c)/(a*d + b*c)  
    #dist = b + c
    #print('distance =', dist)
    return dist                         

# ---------------------------------------------------------------------------
# Creating a matrix of distance between datapoints 
# ---------------------------------------------------------------------------
def distance_matrix(data):
    ''' Function for creating a N X N matrix of distance between datapoints '''
# ---------------------------------------------------------------------------    
    num_data = np.shape(data)[0]
    dmat = np.zeros((num_data, num_data)) 
    
    for row in range(0, num_data):
        dmat[row,row] = 0.0
        d1 = data[row,:]    
        for col in range(row, num_data):                
            d2 = data[col,:]
            d = distance(d1, d2)            
            dmat[row, col] = d
            dmat[col, row] = dmat[row, col]
            
# -----------------------------------------------------------------------------            
    return dmat
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Function for design of experiments 
# -----------------------------------------------------------------------------
def doe(data, initial_index, no_samp): 
    ''' Function for design of experiments ''' 
# Creating a list of indices     
    indices = [None]*no_samp              
# Distance between datapoints               
    dist_mat = distance_matrix(data)              
# First point     
    indices[0] = initial_index
    dist1 =  dist_mat[indices[0], :]       
# Second point    
    indx = np.argmax(dist1) 
    indices[1] = indx
    dist2 =  dist_mat[indices[1], :]
# Minimum distance    
    min_dist = np.minimum(dist1, dist2)    
# -----------------------------------------------------------------------------           
    for i in range(2, no_samp):
        dist1 = min_dist
        indx = np.argmax(dist1) 
        indices[i] = indx
        dist2 =  dist_mat[indices[i], :]
# Minimum distance    
        min_dist = np.minimum(dist1, dist2)    
                
    return indices
               
           
           

# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:53:32 2017

@author: piyush.t
"""

''' Functions for analysis of variance ''' 
import numpy as np
def rsquare(y, f):
    ''' Function for obtaining R^2 of fitted curve ''' 
    #indx = np.where(f[:,0] > 0)[0]
    #f = f[indx,:]; y = y[indx,:]
    return (1.0 - (np.sum(np.square(y-f), axis=0)/(np.sum(np.square(y-np.mean(y, axis=0)), axis=0))))

def rmse(y, f):
    ''' Function for obtaining root mean squared error of fitted curve '''
    #indx = np.where(f[:,0] > 0)[0]
    #f = f[indx,:]; y = y[indx,:]
    return np.sqrt(np.mean(np.square(y - f), axis = 0));

def rmse_accuracy(y, f):
    ''' Function for obtaining rmse accuracy '''
    err = np.sqrt(np.mean(np.square((y - f)/y), axis = 0));
    return (1.0 - err)*100              


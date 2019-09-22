# -------------------------------------------------------------------------
# Functions for visualizing a 2D molecule structure from canonical SMILES 
# -------------------------------------------------------------------------


import numpy as np
import scipy as sc 

from matplotlib import pyplot as plt


def draw_smiles(smiles):
# --------------------------------------------------
# Drawing SMILES 
# --------------------------------------------------
    nimg = len(smiles)    
      
    fig = plt.figure(1)
    nrow = np.int(np.sqrt(np.int(nimg)))   
    ncol = nrow 

    for i in range(0, nrow*ncol):
        ax = fig.add_subplot(nrow*ncol, 1, i+ 1)
        plt.text(0.1, 0.1, smiles[i], transform=ax.transAxes)
        plt.axis('off') 

    plt.show() 

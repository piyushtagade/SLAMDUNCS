# -----------------------------------------------------
# This file contains different drawing tools that can 
# be used for visualization of deep learning results
# -----------------------------------------------------
import numpy as np 
import scipy as sc
import scipy.misc as smp 
#import PIL
#from PIL import Image 


# -----------------------------------------------------
def draw_greyscale(data, irows, icols, asratio): 
# -----------------------------------------------------
# This function draws a greyscale image from a given pixel data
# -----------------------------------------------------
    img_data = np.reshape(data, (irows, icols))
    img = smp.toimage( img_data )
    img = img.resize((img.size[0]*asratio, img.size[1]*asratio), PIL.Image.ANTIALIAS) 
    img.show() 

# -----------------------------------------------------
#   For drawing a stack of images 
# -----------------------------------------------------
def draw_stack_of_images(data, irows, icols, nimg):
# -----------------------------------------------------   

    [num_data, num_dim] = np.shape(data) 
# Aspect ratio     
    asratio = 1 
    
    img_data = np.zeros((irows*asratio*nimg, irows*asratio*nimg))


    for ir in range(0, nimg): 
        for ic in range(0, nimg): 
#            img = draw_greyscale(data[ir*nimg+ic], irows, icols, asratio) 
            img = np.reshape(data[ir*nimg+ic], (irows, icols))
            ir1 = ir*irows; ir2 = (ir+1)*irows
            ic1 = ic*icols; ic2 = (ic+1)*icols 
            img_data[ir1:ir2, ic1:ic2] = img 

    img = smp.toimage( img_data )
    return img.resize((img.size[0]*asratio, img.size[1]*asratio), PIL.Image.ANTIALIAS) 
    
     



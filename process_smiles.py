# -------------------------------------------------------------
#   Functions for processing SMILES data  
# -------------------------------------------------------------

import numpy as np
import scipy as sc
from pprint import pprint 
from design_of_experiments import *
from matplotlib import pyplot as plt
import re 


# -------------------------------------------------------------
# Function to convert SMILES to binary data
# -------------------------------------------------------------
def smiles_to_binary(smiles, max_smiles_length):

# -------------------------------------------------------------
# First checking length of the SMILES
# -------------------------------------------------------------
    smiles_length = len(smiles) 
# 
    
    if smiles_length > max_smiles_length:
         smiles = smiles[0:max_smiles_length] 
         #print('Greater')

    if smiles_length < max_smiles_length:
         rem = max_smiles_length - smiles_length 
         for i in range(0, rem):
              smiles = smiles + ' ' 

    bin_smiles = string_to_binary(smiles)
    return bin_smiles
# --------------------------------------------------------------
# Function to convert string to a list of binary
# --------------------------------------------------------------
def string_to_binary(string): 
    binary = np.zeros([len(string), 8]) 
    for i in range(0, len(string)):
         s = string[i]
         b = bin(ord(s))[2:].zfill(8)
         for j in range(0, 8): 
             binary[i,j] = b[j]  

    return binary.astype(int)


# --------------------------------------------------------------
# Function to convert binary to string 
# --------------------------------------------------------------
def binary_to_string(binary): 
    if binary.ndim > 1:
        smiles_length, bin_length = np.shape(binary)  
    else:
        smiles_length = 1; bin_length = np_shape(binary)[0]    
    string = [] 
    for b in binary:
        s = []
        for j in range(0,8): 
            s.append(str(b[j]))
        string.append(chr(int(''.join(s), 2))) 
    return ''.join(string) 

# -------------------------------------------------------------
# Function to create training and testing dataset  
# -------------------------------------------------------------
def create_smiles_dataset(smiles_data, properties_data=[], max_smiles_length=100, percentage_train_data = 0.7, use_doe = False): 
# ------------------------------------------------------------- 
    total_no_data = len(smiles_data) 
    is_properties = False
    if len(properties_data) > 0:
        is_properties = True 
        properties_data = np.array(properties_data)
    no_train_data = np.int(np.floor(percentage_train_data*total_no_data)) 
    no_test_data = total_no_data - no_train_data


# creating dataset     
    data = [] 
    
    for i in range(0, total_no_data):
        smiles = smiles_data[i]
        b = smiles_to_binary(smiles, max_smiles_length)
        b = np.reshape(b, max_smiles_length*8)
        data.append(b) 
    
    if use_doe == True:
        initial_index = np.random.randint(total_no_data, size = 1)
        indx_train = doe(np.array(data), initial_index, no_train_data)        
    else:
        indx_train = np.random.choice(int(total_no_data), int(no_train_data), replace = False)
        #indx_train = [i for i in range(0, no_train_data)];
# -------------------------------------------------------------
# Training data 
# ------------------------------------------------------------- 
    smiles_batchdata = []; properties_batchdata = [] 
    for i in range(0, no_train_data): 
        j = int(indx_train[i])         
        smiles = smiles_data[j] 
        b = smiles_to_binary(smiles, max_smiles_length)
        b = np.reshape(b, max_smiles_length*8)
        smiles_batchdata.append(b) 
        if is_properties == True:            
            properties_batchdata.append(properties_data[:,j]) 
            #properties_batchdata.append(np.sum(np.array(b))) 
             
            
            
    train_data = (smiles_batchdata, properties_batchdata) 
    
# -------------------------------------------------------------
# Testing data 
# ------------------------------------------------------------- 

   
    #indx_test = set([i for i in range(0, total_no_data)]).symmetric_difference(indx_train)
    indx = [i for i in range(0, total_no_data)];
    indx_test = np.setdiff1d(indx, indx_train)
    print(total_no_data, no_train_data, np.shape(indx_train), np.shape(indx_test))
    no_test_data = np.shape(indx_test)[0]
    smiles_batchdata = []; properties_batchdata = [] 
    for i in range(0, no_test_data):
        j = int(indx_test[i])         
        smiles = smiles_data[j]
        b = smiles_to_binary(smiles, max_smiles_length)
        b = np.reshape(b, max_smiles_length*8)
        smiles_batchdata.append(b) 
        if is_properties == True:
            properties_batchdata.append(properties_data[:,j]) 
    
    test_data = (smiles_batchdata, properties_batchdata) 
    return train_data, test_data  
# -----------------------------------------------------------------
#  For visualization 
# -----------------------------------------------------------------
def visualize_smiles(binary_smiles):
    no_smiles, bin_smiles_length = np.shape(binary_smiles)
    smiles_length = int(bin_smiles_length/8)
    smiles = [] 
    for i in range(0, no_smiles):
        bin_smiles = np.reshape(binary_smiles[i,:], (smiles_length, 8))
        smiles.append(binary_to_string(bin_smiles.astype(int)))
#        pprint(smiles) 

    return smiles
#    draw_smiles(smiles)


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
    plt.pause(0.001)  

# -----------------------------------------------------
# Adding training data to the existing dataset 
# -----------------------------------------------------
def add_data(tot_data, training_data, training_target, test_data, test_target, 
              true_test, test): 

    num_test = np.shape(test_data)[0]

    indx_test = np.array([i for i in range(0, num_test)]) 

    diff = np.square(np.abs(true_test - test))

    diff_oxid = diff[:,2]; diff_red = diff[:,3]

    ids_oxid = np.argsort(diff_oxid); ids_red = np.argsort(diff_red)


    tot_test = len(ids_oxid)
    add_test = np.int(np.floor(tot_data*0.1))

    add_oxid = ids_oxid[tot_test-add_test:tot_test]
    add_red = ids_red[tot_test - add_test : tot_test]

    add_data = np.union1d(add_oxid, add_red)

    rem_testdata = np.setdiff1d(indx_test, add_data)

    add_training_data = test_data[add_data, :]
    tt = np.concatenate((training_data, add_training_data), axis = 0)

    add_training_target = test_target[add_data, :]

    tr = np.concatenate((training_target, add_training_target), axis = 0)

    training_data = tt
    training_target = tr

    test_tt = test_data[rem_testdata, :]
    test_tr = test_target[rem_testdata, :]
    test_data = test_tt
    test_target = test_tr
    return training_data, training_target, test_data, test_target 

# -----------------------------------------------------------------------------
def initialize_smiles(smiles, symmetry=False, side_add = 'both'):
# -----------------------------------------------------------------------------
    '''
        This function returns a smiles for Markov chain initialization and
        associated constraints. 
    '''
# -----------------------------------------------------------------------------
    fixed_index = np.array([], dtype=np.int)    
    copy_loc = [] 
    rand_char = ['C', 'C', '=', 'O', 'C', 'C', 'C', 'C', 'C']    
    
    
    
    i_pah = np.int(len(re.findall(r'\d+', smiles))/2)   
            
    if i_pah > 0:
        r_pah = np.random.choice(2, 1, replace=False)[0] + 1
    else:
        r_pah = 1

    n_pah = np.max(i_pah-r_pah, 0)
    if n_pah < 1:
        n_pah = 1
    #print(n_pah)
    n_pah = i_pah
    nr = 1; #((i_pah + 1) - n_pah)
    #print(nr, i_pah, n_pah)
    #input('HHH')
    ring_index = nr 

    change_index = []
    
    if nr > 1:
       strt = smiles.find(str(nr))-1; en = smiles.rfind(str(nr))+1
       st = smiles[strt:en]       
       st_mod = st[0:2] + st[3:-3] + 'CC=' + st[-2:]
       updated_smiles = smiles[0:strt] + st_mod + smiles[en:]
       
       '''for k in range(0, strt):
           change_index.append(k)
           
       for k in range(en+1, len(updated_smiles)):
           change_index.append(k)'''
           
    else:
       updated_smiles = smiles
    
    print(smiles, updated_smiles)
             
    #print(smiles.find(str(nr)), smiles.rfind(str(nr)))  
         
    #if side_add == 'none':
    
    if symmetry == True: 
       num_sym = np.maximum(np.random.choice(n_pah, 1, replace=False)[0], 1)
       #num_sym = 2
       ret_sym = np.linspace(nr,i_pah,n_pah, dtype=np.int)
       cons_sym = np.random.choice(len(ret_sym), num_sym, replace=False)
       add_char = np.array(np.arange(4,10, dtype=np.int32))    
       #print(ret_sym, cons_sym)
       #ini_insert_loc = []; fin_insert_loc = [] 
       for i_sym in range(0, len(cons_sym)):
           s1 = updated_smiles.find(str(nr))-1; s2 = updated_smiles.rfind(str(nr))+1           
           sym = cons_sym[i_sym]
           print(sym, n_pah, nr)
           n1 = ret_sym[cons_sym[i_sym]]; 
           n2 = ret_sym[(len(ret_sym)-1)-cons_sym[i_sym]]
           if n1 == 1:
                strn = updated_smiles.find(str(n1)) + 2
           elif len(ret_sym) == 1:
                strn = updated_smiles.find(str(n1)) - 1 
           else:
                strn = updated_smiles.find(str(n1)) #- 1 
           if (len(ret_sym)-1)-cons_sym[i_sym] == 0:
                en = updated_smiles.rfind(str(n2)) +1 #2
           else:
                en = updated_smiles.rfind(str(n2))                             
           i_char = np.random.choice(len(add_char), 1, replace=False)[0]
           ic = np.random.choice(len(rand_char), add_char[i_char], replace=True)
           rchar = str('$')
           for i in range(0, add_char[i_char]):
                rchar = rchar + rchar.join(rand_char[ic[i]])           
           rchar = rchar + rchar.join('&')     
           st = updated_smiles[:strn+2] + rchar + updated_smiles[strn+2:en-1] + rchar + updated_smiles[en-1:]           
           updated_smiles = st      
# --------------------------------------------------------------------------------------------------------------------

       if ring_index > 1:
           strt = updated_smiles.find(str(ring_index))-1; en = updated_smiles.rfind(str(ring_index))+1
       
           for k in range(0, strt):
               change_index.append(k)
           
           for k in range(en+1, len(updated_smiles)):
               change_index.append(k)
               
       doller_index = [pos for pos, char in enumerate(updated_smiles) if char == '$']
       ampers_index = [pos for pos, char in enumerate(updated_smiles) if char == '&']
       
       for pos in range(0, len(doller_index)):
           for k in range(doller_index[pos]+1, ampers_index[pos]):
               change_index.append(k)
       updated_smiles = updated_smiles.replace('$', '(')
       updated_smiles = updated_smiles.replace('&', ')')
       
       fixed_index = np.setdiff1d(np.array([i for i in range(0, len(updated_smiles))]), np.array(change_index)) 
       #print(updated_smiles)
       #print(change_index)
       #input('PPP')
       for pos in range(0, np.int(len(doller_index)/2)):
           copy1 = [k for k in range(doller_index[pos]+1, ampers_index[pos])]
           copy2 = [k for k in range(doller_index[np.int(len(doller_index)/2)+(pos)]+1, ampers_index[np.int(len(doller_index)/2)+(pos)])]
           
           copy_loc.append([copy1, copy2])

    if symmetry == False:
       if ring_index > 1:
           strt = updated_smiles.find(str(ring_index))-1; en = updated_smiles.rfind(str(ring_index))+1
       
           for k in range(0, strt):
               change_index.append(k)
           
           for k in range(en+1, len(updated_smiles)):
               change_index.append(k)        
        
        
       add_char = np.array(np.arange(0,5, dtype=np.int32))
    
       fixed_index = np.array([k for k in range(0, len(updated_smiles))])
       i_char = np.random.choice(len(add_char), 1, replace=False)[0]
       ic = np.random.choice(len(rand_char), add_char[i_char], replace=True)
       rchar = ''.join('')
       for i in range(0, add_char[i_char]):
           rchar = rchar + rchar.join(rand_char[ic[i]])
           
               
       if side_add == 'one':
           for k in range(len(updated_smiles), len(updated_smiles)+len(rchar)):
               change_index.append(k)
           updated_smiles = updated_smiles + rchar
           
           
        
        
       if side_add == 'both':
           for i in range(0, len(change_index)):
               change_index[i] = change_index[i] + len(rchar)
               
           for k in range(0, len(rchar)):
               change_index.append(k)
           
           for k in range(len(rchar)+len(updated_smiles), len(updated_smiles)+2*len(rchar)):
               change_index.append(k)
           updated_smiles = rchar + updated_smiles + rchar             
       
        
       if len(change_index) == 0:
           num_change = np.random.choice(len(updated_smiles), 1, replace=False)[0]
           change_loc = np.random.choice(len(updated_smiles)-num_change, 1, replace=False)[0]
           for k in range(0,num_change):
               change_index.append(change_loc+k)
           
           
        
       fixed_index = np.setdiff1d(np.array([i for i in range(0, len(updated_smiles))]), np.array(change_index))

    return updated_smiles, fixed_index, copy_loc
# ----------------------------------------------------------------------------------------------------    
def initialize_smiles_dimer(smiles, rings=1, symmetry=False, add_side = 0):
# -----------------------------------------------------------------------------
    '''
        This function returns a smiles for Markov chain initialization and
        associated constraints. 
    '''
# -----------------------------------------------------------------------------
    fixed_index = np.array([], dtype=np.int)    
    copy_loc = []
    change_index = []  
    rand_char = ['C', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'P', '=', 'C']   

    if rings == 1:
        if symmetry == True:
            n1 = 19; n2 = 37 
            s1 = smiles[:n1]
            s2 = smiles[n1:n2]
            s3 = smiles[n2:]    
            add_char = np.array(np.arange(4,10, dtype=np.int32))      
            i_char = np.random.choice(len(add_char), 1, replace=False)[0]
            ic = np.random.choice(len(rand_char), add_char[i_char], replace=True)
            rchar = str('$')
            for i in range(0, add_char[i_char]):
               rchar = rchar + rchar.join(rand_char[ic[i]])           
            rchar = rchar + rchar.join('&')  
            updated_smiles = s1 + rchar + s2 + rchar + s3
        else:
            updated_smiles = smiles 
# -----------------------------------------------------------------------------------
    else:
# -----------------------------------------------------------------------------------
        if symmetry == True:
            n1 = 4; n2 = 31
            n3 = 13; n4 = 41 
            i_sym = np.random.random(1)
            print('i_sym = ', i_sym) 
            if i_sym < 0.5:
                s1 = smiles[:n1]
                s2 = smiles[n1:n3]
                s3 = smiles[n3:n2]
                s4 = smiles[n2:n4]
                s5 = smiles[n4:]
# -----------------------------------------------------------------------------------
                add_char = np.array(np.arange(2,5, dtype=np.int32))
                i_char = np.random.choice(len(add_char), 1, replace=False)[0]
                ic = np.random.choice(len(rand_char), add_char[i_char], replace=True)
                rchar1 = str('$')
                for i in range(0, add_char[i_char]):
                    rchar1 = rchar1 + rchar1.join(rand_char[ic[i]])
                rchar1 = rchar1 + rchar1.join('&')
# ------------------------------------------------------------------------------------
                add_char = np.array(np.arange(2,5, dtype=np.int32))
                i_char = np.random.choice(len(add_char), 1, replace=False)[0]
                ic = np.random.choice(len(rand_char), add_char[i_char], replace=True)
                rchar2 = str('$')
                for i in range(0, add_char[i_char]):
                    rchar2 = rchar2 + rchar2.join(rand_char[ic[i]])
                rchar2 = rchar2 + rchar2.join('&')
                updated_smiles = s1 + rchar1 + s2 + rchar2 + s3 + rchar1 + s4 + rchar2 + s5
            else:
                if np.random.random(1) < 0.5:
                    s1 = smiles[:n1]
                    s2 = smiles[n1:n2]
                    s3 = smiles[n2:]
                else:
                    s1 = smiles[:n3]
                    s2 = smiles[n3:n4]
                    s3 = smiles[n4:]
                add_char = np.array(np.arange(2,5, dtype=np.int32))
                i_char = np.random.choice(len(add_char), 1, replace=False)[0]
                ic = np.random.choice(len(rand_char), add_char[i_char], replace=True)
                rchar = str('$')
                for i in range(0, add_char[i_char]):
                   rchar = rchar + rchar.join(rand_char[ic[i]])
                rchar = rchar + rchar.join('&')
                updated_smiles = s1 + rchar + s2 + rchar + s3
        else:
            updated_smiles = smiles
# ------------------------------------------------------------------------------------       
        n1 = updated_smiles.find('N') + 1
        n2 = updated_smiles.rfind('N') + 1
# ------------------------------------------------------------------------------------
        add_char = np.array(np.arange(2,5, dtype=np.int32))      
        i_char = np.random.choice(len(add_char), 1, replace=False)[0]
        ic = np.random.choice(len(rand_char), add_char[i_char], replace=True)
        rchar = str('(')
        for i in range(0, add_char[i_char]):
            rchar = rchar + rchar.join(rand_char[ic[i]])           
        rchar = rchar + rchar.join(')')  
# ------------------------------------------------------------------------------------
        if add_side == 1:
            s1 = updated_smiles[:n1]
            s2 = updated_smiles[n1:]
            updated_smiles = s1 + rchar + s2
# ------------------------------------------------------------------------------------
        if add_side == 2:
            s1 = updated_smiles[:n1]
            s2 = updated_smiles[n1:n2]
            s3 = updated_smiles[n2:]          
# ------------------------------------------------------------------------------------
            add_char = np.array(np.arange(2,5, dtype=np.int32))
            i_char = np.random.choice(len(add_char), 1, replace=False)[0]
            ic = np.random.choice(len(rand_char), add_char[i_char], replace=True)
            rchar2 = str('(')
            for i in range(0, add_char[i_char]):
                rchar2 = rchar2 + rchar2.join(rand_char[ic[i]])
            rchar2 = rchar2 + rchar2.join(')')
 
            updated_smiles = s1 + rchar + s2 + rchar2 + s3
# ------------------------------------------------------------------------------------       
        n1 = updated_smiles.find('N')
        n2 = updated_smiles.rfind('N')
        if add_side == 1:
            for k in range(n1+1, n1+len(rchar)):
                change_index.append(k)
# ------------------------------------------------------------------------------------  
        if add_side == 2:
            for k in range(n2+1, n2+len(rchar2)):
                change_index.append(k)     
# ------------------------------------------------------------------------------------
        doller_index = [pos for pos, char in enumerate(updated_smiles) if char == '$']
        ampers_index = [pos for pos, char in enumerate(updated_smiles) if char == '&']
# -------------------------------------------------------------------------------------       
        for pos in range(0, np.int(len(doller_index)/2)):
            copy1 = [k for k in range(doller_index[pos]+1, ampers_index[pos])]
            copy2 = [k for k in range(doller_index[np.int(len(doller_index)/2)+(pos)]+1, ampers_index[np.int(len(doller_index)/2)+(pos)])]
           
            copy_loc.append([copy1, copy2])

        for pos in range(0, len(doller_index)):
            for k in range(doller_index[pos]+1, ampers_index[pos]):
                change_index.append(k)
        updated_smiles = updated_smiles.replace('$', '(')
        updated_smiles = updated_smiles.replace('&', ')')
        fixed_index = np.setdiff1d(np.array([i for i in range(0, len(updated_smiles))]), np.array(change_index))

        print(smiles)
        print(updated_smiles)
        print(change_index)
        print(fixed_index)  

    return updated_smiles, fixed_index, copy_loc
# ----------------------------------------------------------------------------------------

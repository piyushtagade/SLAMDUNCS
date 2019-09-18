import numpy as np
import scipy as sc 
import json 
import urllib.request as ur
import os.path 
from pprint import pprint 
from visualize_mol import draw_smiles
from process_smiles import * 


def get_properties(pid):
### This function reads in cid of the PubChem data and returns a smile
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + str(pid) + '/property/MolecularWeight,ExactMass,CanonicalSMILES/JSON' 
    try:  
         data = ur.urlopen(url).read().decode('utf-8')
         json_data = json.loads(data)
         smiles = json_data['PropertyTable']['Properties'][0]['CanonicalSMILES']
         mass =  json_data['PropertyTable']['Properties'][0]['ExactMass']
         mol_weight = json_data['PropertyTable']['Properties'][0]['MolecularWeight']
         exception = 0
    except:
         smiles = [] 
         mass = []
         mol_weight = []
         exception = 1

    return [exception, smiles, mass, mol_weight]




# this program reads molecule data from PubChem and stores relevent properties in a file. 


def save_pubchem_data(initial_pid, final_pid, fname):  
#    fid = open(fname, 'a')
    all_pids = [i for i in range(initial_pid, final_pid+1)]

    fjson = fname + '.jsn'

    if os.path.isfile(fjson):
         pprint('File Exists')
         with open(fjson, 'r') as fid:
              data = fid.read()  
              json_data = json.loads(data)
              pids = json_data['Properties']['Pid']
              smiles = json_data['Properties']['SMILES']
              mass = json_data['Properties']['MolecularMass']
              mol_weight = json_data['Properties']['MolecularWeight']

    else: 
         pids = []; smiles = []; mass = []; mol_weight = []
    req_pids = set(all_pids).symmetric_difference(pids) 
    for pid in req_pids:
         pprint('PID = ' + str(pid))
         properties = get_properties(pid)
         exception = properties[0]
         if exception == 0: 
              pids.append(pid)
              smiles.append(properties[1])
              mass.append(properties[2])
              mol_weight.append(properties[3]) 
                  
    mol_properties = json.dumps({"Properties":{'Pid': pids, 'SMILES':smiles, 'MolecularMass': mass, 'MolecularWeight': mol_weight}})


    with open(fjson, 'w') as fid:  
         fid.write(mol_properties)

    fdata = fname + '.dat' 

    with open(fdata, 'w') as fid:
         for i in range(0, len(pids)):
              fid.write('{} {} {} {} \n'.format(str(pids[i]), smiles[i], str(mass[i]), str(mol_weight[i])))
    #pprint(mol_properties) 



# -------------------------------------------------------------
# This program reads data from a json file . 
# -------------------------------------------------------------
def read_json_data(fname, redox = False, is_properties = False, max_smiles_length = 1000):  

    fjson = fname + '.jsn'    
    if os.path.isfile(fjson):
         pprint('File Exists')
         with open(fjson, 'rb') as fid:
              #data = fid.read()  
              json_data = json.load(fid)              
         pids = json_data['Properties']['Pid']
         smiles = json_data['Properties']['SMILES']
         if is_properties == True:
              #mass = json_data['Properties']['MolecularMass']
              #mol_weight = json_data['Properties']['MolecularWeight']
              #xlogp = json_data['Properties']['XLogP']
              #tpsa = json_data['Properties']['TPSA']
              #complexity = json_data['Properties']['Complexity']
              #charge = json_data['Properties']['Charge']
              if redox == True:
                   oxid = json_data['Properties']['Oxidation']
                   red = json_data['Properties']['Reduction']
                   homo = json_data['Properties']['homo']
                   lumo = json_data['Properties']['lumo']
         
    else:
         pprint('File does not exists') 
         pids = []; smiles = []; mass = []; mol_weight = []
    lsmiles = len(smiles) 
    #print(lsmiles)
    sm1 = []; mm1 = []; mlw1 = []; ox1 = []; rd1 = []; hm1 = []; lm1 = []  
    xlogp1 = []; tpsa1 = []; cmplx1 = []; chrg1 = [] 
    for i in range(0, lsmiles):
         #print(i)
         if len(smiles[i]) <= max_smiles_length:              
              sm1.append(smiles[i])
              if is_properties == True:
                   #mm1.append(mass[i])
                   #mlw1.append(mol_weight[i]) 
                   #xlogp1.append(xlogp[i])
                   #tpsa1.append(tpsa[i])
                   #cmplx1.append(complexity[i])
                   #chrg1.append(charge[i])
                   if redox == True:
                       ox1.append(oxid[i])
                       rd1.append(red[i])
                       hm1.append(homo[i])
                       lm1.append(lumo[i])
               

    properties = [] 
    
    
    if is_properties == True:
         #properties.append(mm1); properties.append(mlw1); 
         if redox == True:                     
        #properties.append(ox1); properties.append(ox1); properties.append(rd1); properties.append(rd1)             
             properties.append(hm1); properties.append(lm1);                  
             #properties.append(ox1); properties.append(rd1);                               
             #print(min(properties[0]), min(properties[1]), min(properties[2]), min(properties[3]))                 
             #print(max(properties[0]), max(properties[1]), max(properties[2]), max(properties[3]))                 
                              
         #properties.append(xlogp1); properties.append(tpsa1)
         #properties.append(cmplx1); properties.append(chrg1);                  
    #return [smiles, properties]  
    
    return [sm1, properties]  

# -----------------------------------------------------------------------------
# For data pruning 
# -----------------------------------------------------------------------------

def prune(smiles, properties):
    sm1 = []; hm = []; lm = []; ox = []; rd = []; prop = []  
    
    
    homo = np.array(properties[0]); lumo = np.array(properties[1])
    #oxid = np.array(properties[2]); red = np.array(properties[3])
    #log_oxid = np.log(oxid); log_red = np.log(red)
    for i in range(0, len(homo)):
        #print(i, len(oxid), len(smiles))
        if (smiles[i].count('.')==0 and smiles[i].count('Br') == 0 and 
                    smiles[i].count('Os')==0 and smiles[i].count('V')==0 and smiles[i].count('Tb')==0 
                    and smiles[i].count('Ru')==0 and smiles[i].count('Pt')==0 and smiles[i].count('Eu')==0 and smiles[i].count('[')<5 
                              and smiles[i].count('Ti')==0 and smiles[i].count('Zr')==0 and smiles[i].count('Nd')==0 and smiles[i].count('Er')==0
                                        and smiles[i].count('Y')==0 and smiles[i].count('U')==0 and smiles[i].count('Mo')==0 and smiles[i].count('La')==0
                                                  and smiles[i].count('Re')==0):
        # if smiles[i].count('CCC(=O)Cl') > 0:      
            sm1.append(smiles[i]); 
            hm.append(homo[i]); lm.append(lumo[i]) 
            #ox.append(oxid[i]); rd.append(red[i])
            #lox.append(log_oxid[i]); lrd.append(log_red[i])

    prop.append(hm); prop.append(lm); #prop.append(ox); prop.append(rd)
    #prop.append(xl); prop.append(tp); prop.append(compl); prop.append(ch)
    #prop.append(lox); prop.append(lrd)
    return [sm1, prop]


# -----------------------------------------------------------------------------
# For checking duplicate smiles
# -----------------------------------------------------------------------------
def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
  # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set( x for x in seq if x in seen or seen_add(x) )
  # turn the set into a list (as requested)
    return list( seen_twice )



# -----------------------------------------------------------------------------
# Pruning for duplicate smiles
# -----------------------------------------------------------------------------
def prune_dupicates(smiles, properties):
    duplicate_smiles = list_duplicates(smiles)
    pruned_smiles = []; pruned_properties = [] 
    mw = properties[0]; mm = properties[1]; ox = properties[2]; rd = properties[3] 
    mw1 = []; mm1 = []; ox1 = []; rd1 = [];  
    for i in range(0,len(smiles)):
         is_duplicate = 0
         for j in range(0, len(duplicate_smiles)):
             if smiles[i] == duplicate_smiles[j]:
                 is_duplicate = 1
         if is_duplicate == 0:
             pruned_smiles.append(smiles[i])
             mw1.append(mw[i]); mm1.append(mm[i]); ox1.append(ox[i]); rd1.append(rd[i])  
    pruned_properties.append(mw1); pruned_properties.append(mm1); 
    pruned_properties.append(ox1); pruned_properties.append(rd1)
    return pruned_smiles, pruned_properties         

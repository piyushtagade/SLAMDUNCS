# SLAMDUNCS
A deep learning Bayesian framework for attribute driven inverse materials design

Codes for generating the results presented in the manuscript "Attribute driven inverse materials design using deep learning Bayesian framework". 

Steps for code usage:
1) Run bayes_pubchem.py
2) Move files "Generated_smiles_" to the folder "Obtain_unique"
3) Run postprocess_smiles.py
4) Move "Generated_unique_smiles.smi" to the folder "Obtain_smiles"
5) Run validate_structures.py
6) Run get_final_structures.py
7) File "Considered_smiles.smi" contains all the valid molecular structures predicted by SLAMDUNCS.
8) For final screening, use forward prediction to ensure structures exhibit desired attributes. Use DFT to validate properties for selected structures. 

Database used for training and testing the models is available in the files : a) Molecular_data_pubchem.jsn b) IQR_screened_redox_homolumo.jsn

Usage of the code is governed by "Samsung Publication License". 

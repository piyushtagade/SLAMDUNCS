''' 
    Provides a class for Deep Bayesian Infererence.      
''' 
import numpy as np
import scipy as sc
from deep_learning import * 
# ----------------------------------------------------
class Deep_Bayesian_Inference:
    ''' Initializing the class ''' 
    def __init__(self, likelihood, prior): 
        ''' Give details of initialization ''' 
# ---------------------------------------------------
# Functions for likelihood and prior 
        self.likelihood = likelihood 
        self.prior = prior 
# ----------------------------------------------------
# Parameters for MCMC sampling 
        self.number_of_samples = 50000
        self.burnout_period = 10000 
# ---------------------------------------------------- 
# Parameters for annealed sampling 
# Annealing is found to work fine with random beta 
# ---------------------------------------------------- 
        self.ini_fwd_temp  = 0.1
        self.fin_fwd_temp  = 0.9 

        self.ini_back_temp = 0.1
        self.fin_back_temp = 0.9 

        self.anealing_samples = 1000 
# ---------------------------------------------------- 
# Setting options 
# ---------------------------------------------------- 
    def options(self, number_of_samples, burnout_period, 
                initial_fwd_temp = 0.1, final_fwd_temp = 0.9, 
                initial_back_temp = 0.1, final_back_temp = 0.9,
                annealing_samples = 1000): 
# ---------------------------------------------------- 
# Parameters for MCMC sampling 
        self.number_of_samples = number_of_samples
        self.burnout_period = burnout_period
# ---------------------------------------------------- 
# Parameters for annealed sampling 
        self.ini_fwd_temp  = initial_fwd_temp
        self.fin_fwd_temp  = final_fwd_temp 

        self.ini_back_temp = initial_back_temp
        self.fin_back_temp = final_back_temp

        self.annealing_samples = annealing_samples 


# ---------------------------------------------------- 
# Function for Bayesian inference 
# ---------------------------------------------------- 
    def inference(self, data, sdev, ini_samp,  
                  change_index = [], copy_loc = []): 
        ''' Function for Bayesian inference '''
# ----------------------------------------------------        
# Setting prior and likelihood         
# ----------------------------------------------------        
        prior = self.prior
        likelihood = self.likelihood
# ----------------------------------------------------
        if data.ndim == 1:
            nd = np.shape(data)[0]
            data.reshape(1, nd) 
        else:
            nd = np.shape(data)[1] 
        if sdev.ndim > 1:
            sdev = sdev[0,:]   
# ----------------------------------------------------
# Defining covariance matrix 
        cov = np.diag(sdev*sdev)       
# ----------------------------------------------------
# Parameters for annealing 
# ----------------------------------------------------
        '''
        ift = self.ini_fwd_temp; fft = self.fin_fwd_temp 
        ibt = self.ini_back_temp; fbt = self.fin_back_temp
        anneal_samp = self.annealing_samples '''   
# ----------------------------------------------------
# Number of samples 
        number_of_samples = self.number_of_samples 
        burnout_period = self.burnout_period 
# ----------------------------------------------------
# For collecting the samples 
        post_samples = []; accepted_samples = []; visited_pred = []; visited_states = []    
# ----------------------------------------------------          
# Initializing the MCMC sampling 
        pres_state = ini_samp 
        #pres_prior = prior.probability(pres_state)    
# Likelihood for present state 
        pres_pred = likelihood.predict(pres_state) 
       
        
        print(np.shape(pres_pred))  
                               
                               
        homo_max = -4.62512166; homo_min = -8.49539908; homo_xh = -5.6; homo_maxx = homo_max
        lumo_max = 0.70178201; lumo_min = -3.91109452
        slp_homo = 2.0/(homo_maxx + 5.6); cnst_homo = -((homo_maxx - 5.6)/(homo_maxx + 5.6))
        #slp_lumo = 2.0/(lumo_max + 2.15); cnst_lumo = -((lumo_max - 2.15)/(lumo_max + 2.15))
        slp_lumo = 2.0/(-3.15-lumo_min); cnst_lumo = -(( - 3.15 + lumo_min)/(-3.15 - lumo_min))
        
        #slp_homo = 1.0/(homo_xh-homo_max); cnst_homo = -((homo_max )/(homo_xh-homo_max))
        
        homo = (pres_pred[:,0])*(homo_max - homo_min) + homo_min                        
        lumo = (pres_pred[:,1])*(lumo_max - lumo_min) + lumo_min        
        y_homo = (slp_homo*homo + cnst_homo)*10.0
        y_lumo = (slp_lumo*lumo + cnst_lumo)*10.0

        p_homo = (1.0/(1.0+np.exp(-y_homo)))
        p_lumo =  (1.0/(1.0+np.exp(y_lumo)))
        
        #alpha_gamma = 1.0; beta_gamma = 0.5 
        #p_homo = y_homo**(alpha_gamma - 1)*np.exp(-beta_gamma*y_homo)
        
        #if p_homo < 1e-20:
        #    p_homo = 0.0
            
        #if p_lumo < 1e-10:
        #    p_lumo = 0.0
        
        pres_like = np.log(p_lumo) # * p_lumo)
        
        '''if p_homo > 1e-03 and p_lumo > 1e-03:
            
        else:
            pres_like = -1e300 #np.log(1e-300)''' 
        
        print(homo, lumo, y_homo, y_lumo, pres_like)
        #input("PRESS RETURN TO CONTINUE!!!!")                               
# ----------------------------------------------------
# Starting MCMC sampling 

        num_parallel_chains = np.shape(ini_samp)[0]
 
        accepted = np.zeros(num_parallel_chains, dtype=np.int); is_accepted = False   
        homo_state = []; lumo_state = []; visited_pred = []                    
        for samp in range(0, burnout_period+number_of_samples):           
# -----------------------------------------------------  
# Proposed state 
            if samp < 2000:
                beta = 0.2*np.random.random(1)
                tiny_prob = 1e-100

            if samp < 10000 and samp >= 2000:
                beta = 0.3*np.random.random(1)
                tiny_prob = 1e-50   
                
                
            if samp < 15000 and samp >= 10000:
                beta = 0.4*np.random.random(1)
                tiny_prob = 1e-20                

            if samp < 20000 and samp >= 15000:
                beta = 0.5*np.random.random(1)
                tiny_prob = 1e-10                

            if samp < 30000 and samp >= 20000:
                beta = 0.6*np.random.random(1)
                tiny_prob = 1e-7                

                
            if samp > 30000:
                beta = 0.7*np.random.random(1)
                tiny_prob = 1e-5
                                                
            prop_state, prop_ratio, prior_ratio = prior.proposal(pres_state, 
                                                    beta, beta, change_index=change_index, copy_loc = copy_loc)


# Likelihood for proposed state 
            prop_pred = likelihood.predict(prop_state)
            #print(prop_pred)
            #visited_states.append(prop_state)
            
            #smiles = visualize_smiles(np.array(prop_state))
            
            #print(smiles)
            
            '''sigma = cov + np.diag(prop_sd*prop_sd)
            
            x = data - prop_pred
            prop_like = -0.5*np.log(np.linalg.det(sigma)) -0.5*np.dot(np.dot(x, sigma), x.T) '''

            homo = (prop_pred[:,0])*(homo_max - homo_min) + homo_min                        
            lumo = (prop_pred[:,1])*(lumo_max - lumo_min) + lumo_min        
            y_homo = (slp_homo*homo + cnst_homo)*10.0
            y_lumo = (slp_lumo*lumo + cnst_lumo)*10.0
            #prop_like = np.log(((1.0/(1.0+np.exp(-y_homo))))*((1.0/(1.0+np.exp(y_lumo))))) 
            #p_homo = y_homo**(alpha_gamma - 1)*np.exp(-beta_gamma*y_homo)
            p_homo = (1.0/(1.0+np.exp(-y_homo)))
            p_lumo =  (1.0/(1.0+np.exp(y_lumo)))

            visited_pred.append(homo) 
            #print(homo, y_homo, p_homo)

            if p_homo < tiny_prob:
                 p_homo = 0.0
            
            if p_lumo < tiny_prob:
                 p_lumo = 0.0
        
            prop_like = np.log(p_lumo) # * p_lumo)


            '''if p_homo > 1e-03 and p_lumo > 1e-03:
                prop_like = np.log(p_homo * p_lumo)
            else:
                prop_like = -1e300; #np.log(1e-300)''' 
                 
            
            '''oxid = (prop_pred[:,0]/10.0)*(oxid_max - oxid_min) + oxid_min                        
            red = (prop_pred[:,1]/10.0)*(red_max - red_min) + red_min
            redox = oxid - red 
            y_red = (slp_red*red + cnst_red)*6.0
            y_redox = (slp_redox*redox + cnst_redox)*6.0
            prop_like = (1.0/(1.0+np.exp(-y_red)))*(1.0/(1.0+np.exp(-y_redox)))'''                        
# Acceptance probability 
            acceptance_prob = prop_like - pres_like + prop_ratio + prior_ratio
            
            #print(prop_like, pres_like, prop_ratio, prior_ratio)
            
# Metropolis-Hastings criterion
            urand = np.random.random(1)
            '''acceptance_decision = (np.exp(acceptance_prob) > np.random.random(np.shape(acceptance_prob)))

            acc_index = np.where(acceptance_decision >0.99)


            #print(acc_index, np.max(prop_state - pres_state))

            pres_state = pres_state
            pres_like = pres_like
            pres_pred = pres_pred
            acc_pred = pres_pred
            state = pres_state
            is_accepted = False
# --------------------------------------------------------------------
# Changing the accepted states
            accepted[acc_index] = accepted[acc_index] + 1 
            pres_state[acc_index,:] = prop_state[acc_index,:]
            pres_like[acc_index] = prop_like[acc_index]  
            pres_pred[acc_index,:] = prop_pred[acc_index,:]
            state[acc_index,:] = prop_state[acc_index,:]      
            acc_pred[acc_index,:] = prop_pred[acc_index,:]
            is_accepted = True'''
                        


             
            if np.exp(acceptance_prob) > urand:           
# Proposed state accepted 
                accepted = accepted + 1
# --------------------------------------------------------                  
                pres_state = prop_state
                pres_like = prop_like  
                pres_pred = prop_pred
                state = prop_state      
                acc_pred = prop_pred
                is_accepted = True
                
                
            else:  
# --------------------------------------------------------                  
                pres_state = pres_state
                pres_like = pres_like
                pres_pred = pres_pred
                acc_pred = pres_pred
                state = pres_state
                is_accepted = False 
# ---------------------------------------------------------
# Collecting samples after burnout period 
# ---------------------------------------------------------
            homo_state.append(pres_pred[:,0])
            lumo_state.append(pres_pred[:,1])
            
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
                print(samp, accepted[0], homo, lumo) #, np.exp(acceptance_prob), urand) 
 
        return post_samples, accepted_samples, homo_state, lumo_state, visited_pred #, visited_states, visited_pred
# --------------------------------------------------------
 

import math
import numpy as np
import pandas as pd
from diffusion_nmf import DiffusionNMF
from matrix_operation import mat_opr



# Creates a matrix of 0s -- unknown values -- and 1's -- known values ---
# in order to "mask" the data and split it into train/test sets
def train_mask(data, percent_hide):
    # "hide" a given percentage of the data
    num_entries = data.shape[0]*data.shape[1]
    mask = np.zeros(num_entries)
    mask[:int(num_entries * (1 - percent_hide))] = 1
    np.random.shuffle(mask)
    mask = mask.reshape(data.shape)
    
    return mask




# for performing grid search on the parameters
def grid_search(dd, data_mask, laplacian, algorithm, rank_list=range(1,11), beta_list=range(1,11), sparse_list = np.linspace(0,1,11)[5:]):
    I = np.identity(len(laplacian))
    
    # two cases for multiplicative update algorithm or Hoyer projection algorithm
    if algorithm == 'MultUpdate':
        # rows are ranks, columns are beta
        train_err = np.empty((len(rank_list), len(beta_list)))
        test_err = np.empty((len(rank_list), len(beta_list)))
        for rk in rank_list:
            for bt in beta_list:
                kb = np.linalg.inv(I + bt * laplacian)
                differ = DiffusionNMF(dd, kb, M = data_mask, ncomponents = rk, iterations = 1000, tol = 1e-20)
                differ.solver('MultUpdate')
                
                anti_mask = 1 - data_mask
                trn = np.linalg.norm(data_mask * (dd - np.dot(differ.X, np.dot(differ.V, kb)))) / np.linalg.norm(data_mask * dd)
                tst = np.linalg.norm(anti_mask * (dd - np.dot(differ.X, np.dot(differ.V, kb)))) / np.linalg.norm(anti_mask * dd)

                train_err[rk - 1,bt - 1] = trn
                test_err[rk - 1,bt - 1] = tst
                
    else:
        train_err = np.empty((len(rank_list), len(beta_list), len(sparse_list)))
        test_err = np.empty((len(rank_list), len(beta_list), len(sparse_list)))
        for rk in rank_list:
            for bt in beta_list:
                for sp in range(len(sparse_list)):
                    kb = np.linalg.inv(I + bt * state_L)
                    differ = DiffusionNMF(dd, kb, M = data_mask, ncomponents = rk, iterations = 1000, tol = 1e-20)
                    differ.solver('HoyerP', sparseness = sparse_list[sp])
                    
                    anti_mask = 1 - data_mask
                    trn = np.linalg.norm(data_mask * (dd - np.dot(differ.X, np.dot(differ.V, kb)))) / np.linalg.norm(data_mask * dd)
                    tst = np.linalg.norm(anti_mask * (dd - np.dot(differ.X, np.dot(differ.V, kb)))) / np.linalg.norm(anti_mask * dd)
                    train_err[rk - 1,bt - 1, sp] = trn
                    test_err[rk - 1,bt - 1, sp] = tst
    

    return train_err, test_err  






# perfrom average over the grid searches
def avg_train_test(d = None, w = None, h = None, K_true = None, laplacian = None, algorithm = "MultUpdate", test_list = [0.1], runs = 10, 
                        rank_list=range(1,11), beta_list=range(1,11),sparse_list = np.linspace(0,1,11)[5:], saver = False):

    if laplacian is None:
        print("Need laplacian graph for input!")

    if d is None:
        if (not w is None) and (not h is None) and (not K_true is None):
            d = np.dot(w,np.dot(h,K_true))
        else:
            print("Need data matrix D or factors WH for input!")


    results = []
    for t in test_list:
        mult_grid_train_avg = np.nan
        mult_grid_test_avg = np.nan
        
        for r in range(runs):
            training_mask = train_mask(d, t)
            mult_grid_train, mult_grid_test = grid_search(d, data_mask = training_mask, laplacian = laplacian, algorithm = algorithm, 
                                                    rank_list = rank_list, beta_list = beta_list, sparse_list = sparse_list)
            if np.isnan(mult_grid_train_avg).all():
                mult_grid_train_avg = mult_grid_train
                mult_grid_test_avg = mult_grid_test
            else:
                mult_grid_train_avg += mult_grid_train
                mult_grid_test_avg += mult_grid_test
            
        mult_grid_train_avg /= runs
        mult_grid_test_avg /= runs
        tr_min = np.unravel_index(np.argmin(mult_grid_test_avg), mult_grid_test_avg.shape)
        tst_min = np.unravel_index(np.argmin(mult_grid_test), mult_grid_test_avg.shape)

        
        results.append(mult_grid_train_avg)
        results.append(mult_grid_test_avg)
        
    return np.array(results)


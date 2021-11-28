import sys
import os
import numpy as np
import pandas as pd
from parameter_testing import *

cwd = os.getcwd()
par = os.path.dirname(cwd)
sys.path.append(par)

if __name__ == "__main__":
    # import and clean data


    # case counts 
    #state_dset = pd.read_csv(os.path.join(par, 'collected_data/state_dataset.csv'), index_col = 0)
    state_dset = pd.read_csv('./collected_data/state_dataset.csv', index_col = 0)
    state_dset = mat_opr(state_dset)

    # population data
    population = pd.read_csv('./collected_data/state_census_estimate.csv', index_col = 'NAME')

    # clean + normalize
    state_iso = state_dset.known_iso()
    pop_dict = {}
    for col in state_iso.dataframe.columns:
        pop_dict[col] = population.loc[col,'POP']

    state_norm = state_iso.population_normalizer(pop_dict)


    # adjacency Laplacian
    state_L = pd.read_csv("./collected_data/state_laplacian.csv", index_col = 0).to_numpy()

    '''
    W = pd.read_csv('./analysis/testing_data/W_test.csv',index_col=0)
    H = pd.read_csv('./analysis/testing_data/H_test.csv',index_col=0)
    true_beta = 3
    I = np.identity(np.dot(W,H).shape[1])
    true_K = np.linalg.inv(I + true_beta * state_L)
    d = np.dot(W, np.dot(H, true_K))
    '''
    
    b_list = [0.1,0.25,0.5,0.75,1,3,5,7,9]
    error_results = avg_train_test(np.array(state_norm.array), laplacian = state_L, test_list = [0.1, 0.2], runs = 10, 
                        rank_list=range(1,15), beta_list=b_list)

    #error_results = avg_train_test(d, laplacian = state_L, test_list = [0.05, 0.1, 0.2], runs = 10, algorithm="HoyerP")
    
    
    reshaped_error = error_results.reshape(error_results.shape[0], -1)
    np.savetxt("conv_parameter_results.csv", reshaped_error, delimiter = ',')
    f = open("conv_error_shape.txt", "w+")
    f.write(str(error_results.shape))
    f.close()

import sys
import os
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

    # adjacency Laplacian
    state_L = pd.read_csv("./collected_data/state_laplacian.csv", index_col = 0).to_numpy()


    # clean + normalize
    state_iso = state_dset.known_iso()
    pop_dict = {}
    for col in state_iso.dataframe.columns:
        pop_dict[col] = population.loc[col,'POP']
        
    state_norm = state_iso.population_normalizer(pop_dict)

    error_results = avg_train_test(np.array(state_norm.array), laplacian = state_L)
    
    if saver:
        reshaped_error = error_results.reshape(error_results.shape[0], -1)
        np.savetxt("parameter_results.csv", reshaped_error, delimiter = ',')
        f = open("error_shape.txt", "w+")
        f.write(str(error_results.shape))
        f.close()

    print(avg_train_test(np.array(state_norm.array), laplacian = state_L))
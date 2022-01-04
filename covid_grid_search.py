import numpy as np
import pandas as pd
import grid_search
import matrix_operation
from grid_search import gridSearcher
from matrix_operation import mat_opr
import time

###########################################
#
# For performing a grid parameter search over
# our real covid-19 data
#
###########################################

# case counts 
#state_dset = pd.read_csv(os.path.join(par, 'collected_data/state_dataset.csv'), index_col = 0)
state_dset = pd.read_csv('./collected_data/state_dataset.csv', index_col = 0)
state_dset = mat_opr(state_dset)

# population data
population = pd.read_csv('./collected_data/state_census_estimate.csv', index_col = 'NAME')

# clean + normalize
state_iso = state_dset.iso()
pop_dict = {}
for col in state_iso.dataframe.columns:
    pop_dict[col] = population.loc[col,'POP']

state_norm = state_iso.population_normalizer(pop_dict)


# adjacency Laplacian
state_L = pd.read_csv("./collected_data/state_laplacian.csv", index_col = 0).to_numpy()

# grid search over selected list of parameters to find the best
ranks = list(range(1,20))
betas = np.linspace(0.01,5,50)

start = time.time()
G = gridSearcher(state_norm.dataframe, laplacian = state_L, algorithm = "diffusion", max_iter = 20000, tolerance = 1e-8, saver = "./analysis/testing_data/covid_state_grid_search.csv")
G.grid_search(ranks, betas)
end = time.time()

hrs = (end - start) / 60**2
print("made it! Time : " + str(hrs) + " hrs")

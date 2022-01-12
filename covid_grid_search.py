import os
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

# STATE LEVEL
'''
dset = pd.read_csv('./collected_data/state_dataset.csv', index_col = 0)
dset = mat_opr(dset)
#population data
population = pd.read_csv('./collected_data/state_census_estimate.csv', index_col = 'NAME')
# adjacency Laplacian
lapl = pd.read_csv("./collected_data/state_laplacian.csv", index_col = 0).to_numpy()
colname = 'POP'
'''


'''
# COUNTY LEVEL
county_data = pd.read_csv('collected_data/county_dataset.csv', index_col = [0,1,2])
county_data.index = county_data.index.get_level_values("fips")
county_data = county_data.T
dset = mat_opr(county_data)
# county census data for normalization
population = pd.read_csv("./collected_data/county_census.csv", index_col = "fips")
# adjacency laplacian
lapl = pd.read_csv("./collected_data/countyLaplacian.csv", index_col = 0).to_numpy()
colname = 'Population Estimate'
'''

# World Level
world_data = pd.read_csv('collected_data/world_dataset.csv', index_col = 0)
dset = mat_opr(world_data)
population = pd.read_csv('collected_data/world_population.csv', index_col = "Country")
lapl = pd.read_csv('collected_data/worldLaplacian.csv', index_col = 0).to_numpy()
colname = "Population"


# clean + normalize
iso = dset.iso()
pop_dict = {}
for col in iso.dataframe.columns:
    pop_dict[col] = population.loc[col,colname]

norm = iso.population_normalizer(pop_dict)


# grid search over selected list of parameters to find the best
ranks = list(range(1,20))
betas = np.linspace(0,3,40)
iters = 25000
tol = 1e-9
hidden = 0.2
save = "./analysis/testing_data/covid_world_grid_search.csv"

start = time.time()
G = gridSearcher(norm.dataframe, laplacian = lapl, algorithm = "diffusion", max_iter = iters, tolerance = tol, percent_hide = hidden, saver = save)
G.grid_search(ranks, betas)
end = time.time()

hrs = (end - start) / 60**2
print("made it! Time : " + str(hrs) + " hrs")

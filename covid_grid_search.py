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
#dset = mat_opr(dset)
#population data
population = pd.read_csv('./collected_data/state_census_estimate.csv', index_col = 'NAME')
# adjacency Laplacian
lapl = pd.read_csv("./collected_data/state_laplacian.csv", index_col = 0)
colname = 'POP'

lapl = lapl.drop(["Alaska", "Hawaii", "Puerto Rico"], axis = 1)
lapl = lapl.drop(["Alaska", "Hawaii", "Puerto Rico"], axis = 0)
lapl = lapl.to_numpy()

dset = dset.drop(["Alaska", "Hawaii", "Puerto Rico"], axis = 1)
dset = mat_opr(dset)
'''

# COUNTY LEVEL
'''
county_data = pd.read_csv('collected_data/county_dataset.csv', index_col = [0,1,2])
county_data.index = county_data.index.get_level_values("fips")
county_data = county_data.T
dset = mat_opr(county_data)
'''


# county census data for normalization
population = pd.read_csv("./collected_data/county_census.csv", index_col = "fips")
# adjacency laplacian
lapl = pd.read_csv("./collected_data/countyLaplacian.csv", index_col = 0).to_numpy()
colname = 'Population Estimate'

# For smaller portions of counties
lapl = pd.read_csv("./collected_data/countyLaplacian.csv", index_col = 0)
lapl.columns = lapl.columns.astype("int")


# California Counties

cali = pd.read_csv('collected_data/county_dataset.csv', index_col = [0,1,2])
cali = cali.loc[cali.index.get_level_values("state").isin(["California"])]
cali.index = cali.index.get_level_values("fips")
cali = cali.T
dset = mat_opr(cali)

lapl = lapl.loc[cali.columns,cali.columns].to_numpy()
population = population.loc[cali.columns, :]


# New England Columns
'''
new_eng = pd.read_csv('collected_data/county_dataset.csv', index_col = [0,1,2])
new_eng = new_eng.loc[new_eng.index.get_level_values("state").isin(["New York", "Connecticut", "Maine", "Vermont",
                                                       "Massachusetts", "New Hampshire", "Rhode Island"])]
new_eng.index = new_eng.index.get_level_values("fips")
new_eng = new_eng.T
dset = mat_opr(new_eng)

lapl = lapl.loc[new_eng.columns,new_eng.columns].to_numpy()
population = population.loc[new_eng.columns, :]
'''


'''
# World Level
world_data = pd.read_csv('collected_data/world_dataset.csv', index_col = 0)
dset = mat_opr(world_data)
population = pd.read_csv('collected_data/world_population.csv', index_col = "Country")
lapl = pd.read_csv('collected_data/worldLaplacian.csv', index_col = 0).to_numpy()
colname = "Population"
'''


# clean + normalize
iso = dset.iso()
pop_dict = {}
for col in iso.dataframe.columns:
    pop_dict[col] = population.loc[col,colname]

norm = iso.population_normalizer(pop_dict)


# grid search over selected list of parameters to find the best
ranks = list(range(1,20))
#betas = np.linspace(0,2,10)
betas = [1]
iters = 100000
tol = 1e-9
hidden = 0.2
save1 = "./analysis/testing_data/california_diff.csv"
save2 = "./analysis/testing_data/california_nmf.csv"

start = time.time()
G = gridSearcher(norm.dataframe, laplacian = lapl, algorithm = "diffusion", max_iter = iters, validate = 10, tolerance = tol, percent_hide = hidden, saver = save1)
G.grid_search(ranks, betas)

G = gridSearcher(norm.dataframe, laplacian = lapl, algorithm = "nmf", max_iter = iters, validate = 10, tolerance = tol, percent_hide = hidden, saver = save2)
G.grid_search(ranks, betas)

end = time.time()
hrs = (end - start) / 60**2
print("made it! Time : " + str(hrs) + " hrs")


'''
ranks1 = list(range(1,5))
betas1 = np.linspace(0,5,10)
save1 = "./analysis/testing_data/covid_county_grid_search1.csv"
G = gridSearcher(norm.dataframe, laplacian = lapl, algorithm = "diffusion", max_iter = iters, tolerance = tol, percent_hide = hidden, saver = save1)
G.grid_search(ranks1, betas1)


ranks2 = list(range(5,10))
betas2 = np.linspace(0,5,10)
save2 = "./analysis/testing_data/covid_county_grid_search2.csv"
G = gridSearcher(norm.dataframe, laplacian = lapl, algorithm = "diffusion", max_iter = iters, tolerance = tol, percent_hide = hidden, saver = save2)
G.grid_search(ranks2, betas2)


ranks3 = list(range(10,15))
betas3 = np.linspace(0,5,10)
save3 = "./analysis/testing_data/covid_county_grid_search3.csv"
G = gridSearcher(norm.dataframe, laplacian = lapl, algorithm = "diffusion", max_iter = iters, tolerance = tol, percent_hide = hidden, saver = save3)
G.grid_search(ranks3, betas3)

ranks4 = list(range(15,20))
betas4 = np.linspace(0,5,10)
save4 = "./analysis/testing_data/covid_county_grid_search4.csv"
G = gridSearcher(norm.dataframe, laplacian = lapl, algorithm = "diffusion", max_iter = iters, tolerance = tol, percent_hide = hidden, saver = save4)
G.grid_search(ranks4, betas4)
'''



import pandas as pd
import numpy as np
import grid_search
from grid_search import gridSearcher
from data_generator import gen_decomposition, gen_laplacian
import time

# a quick program used for running tests on different randomly generated datasets

# true parameters to define
n = 200
m = 50
rank = 5
beta = 5
random_state = 1729

# generate data using true paramters 
W,H = gen_decomposition(n,m,rank, state=random_state)
laplacian = gen_laplacian(m, state=random_state)
K = np.linalg.inv(np.identity(m) + beta * laplacian)
D = W @ H @ K

# grid search over selected list of parameters to find the best
ranks = list(range(1,25))
betas = np.linspace(1,10,100)

start = time.time()
G = gridSearcher(D, laplacian, saver = "./analysis/testing_data/grid_search_" + str(rank) + "_" + str(beta) + "_" + ".csv")
G.grid_search(ranks, betas)
end = time.time()

hrs = (end - start) / 60**2
print("made it! Time : " + str(hrs) + " hrs")

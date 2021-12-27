import pandas as pd
import numpy as np
import grid_search
from grid_search import gridSearcher
from data_generator import gen_decomposition, gen_laplacian, add_noise
import time

# a quick program used for running tests on different randomly generated datasets

# true parameters to define
n = 200
m = 50
rank = 5
beta = 1
random_state = 1729

# generate data using true paramters 
W,H = gen_decomposition(n,m,rank, state=random_state)
G, laplacian = gen_laplacian(size = m, H = H, p_edge = 0.05, state=random_state)
K = np.linalg.inv(np.identity(m) + beta * laplacian)
D = np.dot(W, np.dot(H, K))
noise_D = add_noise(D, 0.001)

# grid search over selected list of parameters to find the best
ranks = list(range(1,20))
betas = np.linspace(0.001,2,50)

start = time.time()
G = gridSearcher(noise_D, laplacian, saver = "./analysis/testing_data/grid_search_" + str(rank) + "_" + str(beta) + ".csv")
G.grid_search(ranks, betas)
end = time.time()

hrs = (end - start) / 60**2
print("made it! Time : " + str(hrs) + " hrs")

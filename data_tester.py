import pandas as pd
import numpy as np
import grid_search
from grid_search import gridSearcher


X = pd.read_csv("./analysis/testing_data/D_4.csv", index_col = 0).to_numpy()

state_L = pd.read_csv("./collected_data/state_laplacian.csv", index_col = 0).to_numpy()

G = gridSearcher(X, state_L, saver = "./analysis/testing_data/test_search.csv")
ranks = list(range(1,3))
betas = list(range(1,2))
G.grid_search(ranks, betas)

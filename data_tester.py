import pandas as pd
import numpy as np
import grid_search
from grid_search import gridSearcher


X = pd.read_csv("./analysis/testing_data/D_4.csv", index_col = 0).to_numpy()

state_L = pd.read_csv("./collected_data/state_laplacian.csv", index_col = 0).to_numpy()

G = gridSearcher(X, state_L, saver = "./analysis/testing_data/test_search2.csv")
ranks = list(range(1,50))
betas = np.linspace(0.0001,2,31)
G.grid_search(ranks, betas)

print("made it!")

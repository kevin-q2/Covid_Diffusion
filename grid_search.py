import numpy as np 
import pandas as pd
from joblib import Parallel, delayed
import diffusionNMF
from diffusionNMF import diffusionNMF


class gridSearcher:
    def __init__(self, X, laplacian, max_iter = 100000, tolerance = 1e-9, percent_hide = 0.2, validate = 5):
        self.X = X
        self.laplacian = laplacian
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.percent_hide = percent_hide
        self.validate = validate

    def kernelize(self, beta):
        I = np.identity(self.laplacian.shape[0])
        return np.linalg.inv(I + beta * self.laplacian)
    
    def train_mask(self):
        # "hide" a given percentage of the data
        num_entries = self.X.shape[0]*self.X.shape[1]
        mask = np.zeros(num_entries)
        mask[:int(num_entries * (1 - self.percent_hide))] = 1
        np.random.shuffle(mask)
        mask = mask.reshape(self.X.shape)

        return mask
    
    def relative_error(self, W, H, K, mask):
        error = np.linalg.norm(mask * (self.X - np.dot(W, np.dot(H, K))))
        baseline = np.linalg.norm(mask * self.X)
        return error/baseline
    
    def param_solver(self, rank, beta):
        K = self.kernelize(beta)
        M = self.train_mask()
        dSolver = diffusionNMF(n_components = rank, kernel = K, mask = M, n_iter = self.max_iter, tol = self.tolerance)
        W,H = dSolver.fit_transform(self.X)
        
        rel_error = self.relative_error(W,H,K,M)
        return rel_error
    
    
    def grid_search(self, rank_list, beta_list):
        trials = []
        
        for r in rank_list:
            for b in beta_list:
                for v in range(self.validate):
                    trials.append((r,b))
                
                
        res = Parallel(n_jobs = -1)(delayed(self.param_solver)(r,b) for (r,b) in trials)
        
        return res
    

    
import numpy as np 
import pandas as pd
from joblib import Parallel, delayed
import diffusionNMF
import nmf
from diffusionNMF import diffusionNMF
from nmf import nmf


class gridSearcher:
    def __init__(self, X, laplacian = None, algorithm = "diffusion", max_iter = 100000, tolerance = 1e-9, percent_hide = 0.2, noise = None, validate = 5, saver = None):
        self.X = X
        self.laplacian = laplacian
        self.algorithm = algorithm 
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.percent_hide = percent_hide
        self.noise = noise
        self.validate = validate
        self.saver = saver

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
    
    def add_noise(self, data, std_dev = None):
        matr = np.matrix.copy(data)
        if std_dev is None:
            return matr

        else:
            for rower in range(matr.shape[0]):
                for coler in range(matr.shape[1]):
                    noisy = np.random.normal(scale = std_dev)
                    if matr[rower, coler] + noisy < 0:
                        matr[rower, coler] = 0
                    else:
                        matr[rower, coler] += noisy
            
            return matr
    
    def relative_error(self, W, H, K, mask):
        if self.algorithm == "nmf":
            error = np.linalg.norm((1 - mask) * (self.X - np.dot(W,H)))
            baseline = np.linalg.norm((1 - mask) * self.X)
        else:
            error = np.linalg.norm((1 - mask) * (self.X - np.dot(W, np.dot(H, K))))
            baseline = np.linalg.norm((1 - mask) * self.X)
            
        return error/baseline
    
    def param_solver(self, rank, beta):
        M = self.train_mask()
        K = None
        noisy = self.add_noise(self.X, self.noise)
        
        if self.algorithm == "nmf":
            nSolver = nmf(n_components = rank, mask = M, n_iter = self.max_iter, tol = self.tolerance)
            W,H = nSolver.fit_transform(noisy)
        else:
            K = self.kernelize(beta)
            dSolver = diffusionNMF(n_components = rank, kernel = K, mask = M, n_iter = self.max_iter, tol = self.tolerance)
            W,H = dSolver.fit_transform(noisy)
        
        rel_error = self.relative_error(W,H,K,M)
        return rank, beta, rel_error
    
    
    def post_process(self, results):
        res_frame = pd.DataFrame(results)
        res_frame.columns = ["rank", "beta", "relative error"]
        res_frame = res_frame.groupby(["rank","beta"], as_index = False).mean()
        
        if not self.saver is None:
            res_frame.to_csv(self.saver)
            
        return res_frame
    
    def grid_search(self, rank_list, beta_list):
        trials = []
        
        for r in rank_list:
            for b in beta_list:
                for v in range(self.validate):
                    trials.append((r,b))
                
                
        res = Parallel(n_jobs = -1)(delayed(self.param_solver)(r,b) for (r,b) in trials)
        
        return self.post_process(res)
    

    
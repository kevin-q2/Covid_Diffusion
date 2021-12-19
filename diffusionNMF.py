import math
import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import random






def MultUpdate(D, X, V, K, mask):
    # Update based on standard multiplicative update rules

    # Adjust for Masked Values:
    masked_d = mask * D
    masked_xvk = mask * np.dot(X, np.dot(V, K))
    
    # Update X
    num_x = np.dot(masked_d, np.dot(K.T, V.T))
    denom_x = np.dot(masked_xvk, np.dot(K.T, V.T)) + 1e-9
    grad_x = np.divide(num_x, denom_x)
    X = np.multiply(X, grad_x)
    
    
    # Update V
    masked_xvk = mask * np.dot(X, np.dot(V, K))
    num_v = np.dot(X.T, np.dot(masked_d, K.T))
    denom_v = np.dot(X.T, np.dot(masked_xvk, K.T)) + 1e-9
    grad_v = np.divide(num_v, denom_v)
    V = np.multiply(V, grad_v)
    
    return X, V


def solver(X, W, H, K, mask, max_iter, tol):
    # initial objective (cost function)
    O = np.linalg.norm(mask * (X - np.dot(W, np.dot(H, K))))

    iteri = 0

    while iteri < max_iter:
        old_dist = O

        W, H = MultUpdate(X, W, H, K, mask)

        # calculate change in cost and return if its small enough
        O = np.linalg.norm(mask * (X - np.dot(W, np.dot(H, K))))
        change = abs(old_dist - O)

        if change < tol:
            break

        iteri += 1
        
    
    return W, H, iteri


class diffusionNMF(TransformerMixin, BaseEstimator):
    def __init__(self, n_components = None, kernel = None, mask = None, n_iter = 500, tol = 1e-10, progress = False):
        self.n_components = n_components
        self.kernel = kernel
        self.mask = mask
        self.n_iter = n_iter
        self.tol = tol
        self.progress = progress
        
        
    def _check_params(self, X):
        # method to check all initial input parameters
        
        # Check input data
        if X.min() < 0:
            raise ValueError("all elements of input data must be positive")
        
        # Check Kernel input
        if self.kernel is None:
            raise ValueError("Need to provide diffusion kernel")
        else:
            try:
                self.kernel = np.array(self.kernel)
            except:
                raise ValueError('Input array must be 2d numpy array or similar')
            
            if self.kernel.shape[0] != self.kernel.shape[1]:
                raise ValueError('Diffusion Kernel must be a square matrix')
            elif self.kernel.shape[0] != X.shape[1]:
                print(self.kernel.shape)
                print(X.shape)
                raise ValueError("Size of diffusion kernel must match the size of the data's features")
            
        
        # check mask input:
        if self.mask is None:
            self.mask = np.ones(X.shape)
        else:
            if self.mask.shape != X.shape:
                raise ValueError("Input mask must match the size of the data")
        
        
        # Check rank parameter 
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError("Rank must be a positive integer")
        
        # check iterations 
        if not isinstance(self.n_iter, int) or self.n_iter <= 0:
            raise ValueError("Number of iterations must be a positive integer")
        
        
        # check tolerance level
        if not isinstance(self.tol, float) or self.tol <= 0:
            raise ValueError("Tolerance level must be positive floating point value")
            
            
        # Check progress parameter
        if not isinstance(self.progress, bool):
            raise ValueError("Progress parameter must be boolean value")
        
        

        
    def check_w_h(self, X, W, H):
        if W is None:
            W = np.random.rand(len(X), self.n_components)
        else:
            if W.shape != (len(X), self.n_components):
                raise ValueError("W input should be of size " + str((len(X), self.n_components)))
            
        if H is None:
            H = np.random.rand(self.n_components, X.shape[1])
        else:
            if H.shape != (self.n_components, X.shape[1]):
                raise ValueError("H input should be of size " + str((self.n_components, X.shape[1])))
            
        return W,H
        
        
        
    def fit_transform(self, X, y=None, W = None, H = None):
        # initialize X and V
        
        #X = self._validate_data(X, dtype=[np.float64, np.float32])
        try:
            X = np.array(X)
        except:
            raise ValueError("Input data must be array-like")
        
        W,H = self.check_w_h(X, W, H)
        
        self._check_params(X)
        
        
        W,H, n_iters = solver(X, W, H, self.kernel, self.mask, self.n_iter, self.tol)
        
        if n_iters == self.n_iter:
            print("Max iterations reached, increase to converge on given tolerance")
        
        return W,H
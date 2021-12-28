import numpy as np
import pandas as pd


#####################################################################################################################
# A simple Non Negative Matrix Factorization implementation
# Author: Kevin Quinn 10/21/21

# Update rules based on the results established in https://arxiv.org/pdf/1612.06037.pdf

# Specifically, this implementation takes into account that not all values are known,
# and masks the known/unknown values to account for this (More details in the paper)

# INPUT:

#     X - Initial n x m data matrix (with 0s or some other float value to represent missing values)

#     ncomponents - The desired rank k to be used in Factorization

#     M - data mask (1's for known values and 0's for unknown)

#     iterations - desired number of iterations to complete before convergence

#     tol - value at which we decide the change in cost function is small enough to declare convergence 

# OUTPUT:
#     W - n x k factor matrix
#     H - k x m factor matrix

####################################################################################################################

class nmf:
    def __init__(self, X, ncomponents, M = None, iterations = 200, tol = 1e-10, w_init = None, h_init = None):
        if np.any(np.array(X) < 0):
            raise ValueError('Input array is negative')

        self.X = np.array(X)
        self.ncomponents = ncomponents
        if M is None:
            self.M = np.ones(np.shape(X))
        else:
            self.M = np.array(M)
        self.iterations = iterations
        self.tol = tol
        self.W = w_init
        self.H = h_init




    def MultUpdate(self):
        # Update based on standard multiplicative update rules + adjusting for masked values

        # Mask unknown values if any
        masked_X = np.multiply(self.M, self.X)
        masked_WH = np.multiply(self.M, np.dot(self.W, self.H))

        # Update W
        num_w = np.dot(masked_X, self.H.T)
        denom_w = np.dot(masked_WH, self.H.T) + 1e-9 # add 1e-9 to make sure no 0s in the denominator
        lrw = np.divide(num_w, denom_w)
        self.W = np.multiply(self.W, lrw)

        # re-apply mask after updating W
        masked_WH = np.multiply(self.M, np.dot(self.W, self.H))

        # Update H
        num_h = np.dot(self.W.T, masked_X)
        denom_h = np.dot(self.W.T, masked_WH) + 1e-9 
        lrh = np.divide(num_h, denom_h)
        self.H = np.multiply(self.H, lrh)


    def solver(self):
        samples = self.X.shape[0]
        features = self.X.shape[1]

        # initialize W and H
        if self.W is None:
             self.W = np.random.rand(samples, self.ncomponents)

        if self.H is None:
            self.H = np.random.rand(self.ncomponents, features)

        # initial objective (cost function)
        O = np.linalg.norm((self.M ** (1/2)) * (self.X - np.dot(self.W, self.H)))

        # iterate and update W and H
        iteri = self.iterations
        while iteri > 0:
            #w_old = self.W 
            #h_old = self.H
            old_dist = O

            self.MultUpdate()

            # Compute change in overall cost function and return if its small enough
            O = np.linalg.norm((self.M ** (1/2)) * (self.X - np.dot(self.W, self.H)))
            change = abs(old_dist - O)
            if change < self.tol:
                return

            iteri -= 1


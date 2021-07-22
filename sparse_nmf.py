import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.sparse import random
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import scipy.stats as stats


###################################################################################################################################

# A basic implementation of NMF with sparseness constraints
# Original algorithm designed by Patrik Hoyer : https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf
# And the original MATLAB implementation can be found here: https://github.com/aludnam/MATLAB/blob/master/nmfpack/code/projfunc.m

# This is a translation of that code into python for research purposes


# INPUT:
# X -- the original (non-negative) data matrix
# ncomponents -- number of desired basis components for factorization
# sW -- sparseness of W in range [0,1] (leave uninitialized for no constraint)
# sH -- sparseness of H in range [0,1] (leave uninitialized for no constraint)

# Note: a 1 for sW or sH means totally sparse, 0 means not sparse at all

###################################################################################################################################


class SparseNMF:
    def __init__(self, X, ncomponents, iterations = 500, sW = None, sH = None, tol = 1e-10):
        if np.any(np.array(X) < 0):
            raise ValueError('Input array is negative')

        self.X = X
        self.ncomponents = ncomponents
        self.iterations = iterations
        self.sW = sW
        self.sH = sH
        self.tol = tol
    


    def projection(self, s, k1, k2):
        # This method projects a vector s into its closest non-negative counterpart
        # More specifically: it "finds a vector v having sum(abs(v)) = k1 (L1 norm)
        # and sum (v^2) = k2 (L2 norm) which is closest to s in the euclidean sense...restricted to being non-negative"

        # More details can be found within the original paper

        N = len(s)

        # project to the sum constraint:
        v = s + (k1 - sum(s))/N

        zerocoeff = []

        while True:

            # midpoint calculation to help with projection
            midpoints = []
            for i in range(len(s)):
                if i in zerocoeff:
                    midpoints.append(0)
                else:
                    midpoints.append(k1/(N - len(zerocoeff)))

            midpoints = np.array(midpoints)
            w = v - midpoints


            
            a = sum(w**2)
            b = 2*np.dot(w, v)  
            c = sum(v**2) - k2
            '''
            a = sum(w**2)
            b = 2*np.dot(midpoints, w)
            c = sum(midpoints**2)  - k2
            '''

            roots  = np.roots([a,b,c])

            if np.isreal(roots[0]) and np.isreal(roots[1]):
                alpha = max(roots)
            elif np.isreal(roots[0]):
                alpha = roots[0]
            elif np.isreal(roots[1]):
                alpha = roots[1]
            else:
                alpha = max(np.real(roots))


            # project to closest point on the joint constraint hypershpere
            v = alpha*w + v
            #v = midpoints + alpha * w

            # if our new vector is non-negative then we are dont
            negs = v < 0
            if np.any(negs) == False:
                break

            # if not set all negative values to 0 and adjust s
            # also need to record where the zeros are:
            for bol in range(len(negs)):
                if negs[bol] == True and bol not in zerocoeff:
                    zerocoeff.append(bol)

            v[zerocoeff] = 0
            c = (sum(v) - k1)/(N - len(zerocoeff))
            v = v - c
            v[zerocoeff] = 0

        return v

    
    def solver(self):

        #self.scaler = self.X.max()
        #self.X = self.X/self.scaler

        vdim = self.X.shape[0]
        samples = self.X.shape[1]

        # Initialize W and H as random matrices
        self.W = np.random.rand(len(self.X), self.ncomponents)
        self.H = np.random.rand(self.ncomponents, len(self.X[0]))

        # Normalize the L2 norm for each row of H
        self.H = normalize(self.H, norm = 'l2', axis = 0) # NOT SURE IF I CAN just use the scipy normalize or not ...

        # Adjust for sparseness 
        # (based on the sparseness equation in the paper)
        if self.sW is not None:
            L1a = math.sqrt(vdim) - (math.sqrt(vdim) - 1)*self.sW
            for i in range(self.W.shape[1]):
                self.W[:,i] = self.projection(self.W[:,i], L1a, 1)

        if self.sH is not None:
            L1s = math.sqrt(samples) - (math.sqrt(samples) - 1)*self.sH
            for i in range(self.H.shape[0]):
                self.H[i,:] = self.projection(self.H[i,:], L1s, 1)


        # initial objective (cost function)
        #O = 0.5 * ((self.X - np.dot(self.W,self.H)) ** 2).sum()
        O = np.linalg.norm(self.X - np.dot(self.W, self.H))

        # initial step sizes
        w_step = 1
        h_step = 1

        iteri = self.iterations
        while iteri > 0:
            # save previous values
            w_old = self.W 
            h_old = self.H


            # Update H    
            if self.sH is not None:
                dH = np.dot(np.transpose(self.W), np.dot(self.W,self.H) - self.X)
  
                # gradient descent until we've taken a sufficient step
                while True:
                    # gradient
                    h_new = self.H - h_step * dH
                    # project to the l1 norm determined by sparseness
                    for r in range(self.H.shape[0]):
                        h_new[r,:] = self.projection(h_new[r,:], L1s, 1)

                    # check if we've improved the distance
                    #dist = 0.5 * ((self.X - np.dot(self.W,h_new)) ** 2).sum()
                    dist = np.linalg.norm(self.X - np.dot(self.W, h_new))

                    if dist <= O:
                        # if we have then break
                        O = dist
                        h_step = h_step * 1.2
                        self.H = h_new
                        break

                    # else decrease the stepsize and try again
                    else:
                        h_step = h_step / 2

                    # If the stepsize is small enough then we've converged on a solution
                    if h_step < 1e-200:
                        print("H Algorithm Converged")
                        return


            else:
                # standard NMF update step
                # because H is not constrained
                num = np.multiply(self.H, np.dot(np.transpose(self.W), self.X))
                denom = np.dot(np.transpose(self.W), np.dot(self.W, self.H)) + 1e-9
                self.H = np.divide(num, denom)

                # re - normalize
                norms = (sum(np.transpose(self.H) ** 2)) ** 0.5 # l2 norm along rows of H

                # rows of H keep constant energy 
                # W is scaled up to accomodate
                for n in range(len(norms)):
                    self.H[n,:] /= norms[n]
                    self.W[:,n] *= norms[n]

                #O = 0.5 * ((self.X - np.dot(self.W,self.H)) ** 2).sum()
                O = np.linalg.norm(self.X - np.dot(self.W, self.H))



            # Update W
            if self.sW is not None:
                # gradient step
                dW = np.dot(np.dot(self.W, self.H) - self.X, np.transpose(self.H))

                # loop until we decrease the objective
                while True:
                    w_new = self.W - w_step*dW
                    l2 = (sum(w_new ** 2)) ** 0.5 # columnwise l2 norm

                    # project to l1 sparseness and keep original l2 
                    for c in range(self.ncomponents):
                        w_new[:,c] = self.projection(w_new[:,c], L1a*l2[c], l2[c]**2)

                    # check if we've improved the distance
                    #dist = 0.5 * ((self.X - np.dot(w_new,self.H)) ** 2).sum()
                    dist = np.linalg.norm(self.X - np.dot(w_new, self.H))

                    if dist <= O:
                        self.W = w_new
                        O = dist
                        w_step = w_step * 1.2
                        break

                    else:
                        w_step /= 2

                    
                    if w_step < 1e-200:
                        print("W Algorithm Converged")
                        return
            else:
                # standard NMF update step
                # because W is not constrained
                num = np.multiply(self.W, np.dot(self.X, np.transpose(self.H)))
                denom = np.dot(self.W, np.dot(self.H, np.transpose(self.H))) + 1e-9
                self.W = np.divide(num, denom)

                #O = 0.5 * ((self.X - np.dot(self.W,self.H)) ** 2).sum()
                O = np.linalg.norm(self.X - np.dot(self.W, self.H))

            iteri -= 1
        
        #self.W = self.W * self.scaler


    def hoyer_solver(self, lambda_sparse):
        # solves based on Hoyer's original paper on sparse coding 
        # https://arxiv.org/pdf/cs/0202009.pdf
        # But slightly modified to include the diffusion Kernel
        # the parameter lambda_sparse controls the level of sparseness in the output
        vdim = self.X.shape[0]
        samples = self.X.shape[1]


        self.W = np.random.rand(len(self.X), self.ncomponents)
        self.H = np.random.rand(self.ncomponents, len(self.X[0]))

        O = np.linalg.norm(self.X - np.dot(self.W, self.H))
        iteri = self.iterations
        w_step = 1
        while iteri > 0:
            w_old = self.W 
            h_old = self.H
            old_dist = O

            # Update X by modifying the step size until cost decreases
            dW = np.dot(self.X, self.H.T) - np.dot(self.W, np.dot(self.H, self.H.T))

            # loop until we decrease the objective
            while True:
                w_new = self.W + w_step*dW
                
                # set negative values to 0
                # and normalize columns of x
                #w_new[w_new < 0] = 0
                #w_new = normalize(w_new, axis = 0)

                # check if we've improved the distance
                #dist = 0.5 * ((self.X - np.dot(w_new,self.H)) ** 2).sum()
                dist = np.linalg.norm(self.X - np.dot(w_new, self.H))

                if dist <= O:
                    self.W = w_new
                    O = dist
                    w_step = w_step * 1.2
                    break

                else:
                    w_step /= 2

                    if w_step < 1e-100:
                        break

            self.W[self.W < 0] = 0
            #self.W = normalize(self.W, norm = 'l1', axis = 0)
            # Update V
            # Multiplicative update step modified to include sparseness
            # because H is not constrained
            num = np.multiply(self.H, np.dot(self.W.T, self.X))
            denom = np.dot(self.W.T, np.dot(self.W, self.H)) + lambda_sparse + 1e-9 # add 1e-9 to make sure its not 0?
            self.H = np.divide(num, denom)
            #self.H = normalize(self.H, norm = 'l2',axis = 1)


            O = np.linalg.norm(self.X - np.dot(self.W, self.H))
            change = abs(old_dist - O)
            if change < self.tol:
                return

            iteri -= 1








############################################
# TESTING


def gen_data():
    # generates testing data and saves it so I can test in matlab as well
    test_h = random(4,100, density = 0.25).A
    H = pd.DataFrame(test_h)

    x = np.linspace(1,10,100)
    y1 = stats.norm.pdf(x, 5) * 20
    y2 = []
    y3 = []
    y4 = []
    for i in x:
        y2.append((i - 5)**2)
        y3.append(2 * i)
        y4.append((i - 5)**3/10 + 10)
    y4 = np.array(y4)

    W = pd.DataFrame(np.array([y1, y2, y3, y4]).T)

    H.to_csv("H_test.csv")
    W.to_csv("W_test.csv")

    return W,H

if __name__ == '__main__':
    comps = 4
    #W, H = gen_data()
    W = pd.read_csv('W_test.csv', index_col = 0)
    H = pd.read_csv('H_test.csv', index_col = 0)

    X = np.dot(W, H)

    # sklearn decomposition: 
    model = NMF(n_components = comps, init = 'random', random_state = 35)
    sciW = model.fit_transform(X)
    sciH = model.components_

    # sparse decomposition
    # Initial test with 0 sparseness
    sparseness = 0.7

    model2 = SparseNMF(X, comps, iterations = 2000, sW = 0.1, sH = sparseness)    
    model2.solver()
    spW = model2.W
    spH = model2.H

    print("Using sparseness value of:", sparseness)
    print("Scipy error on H:", np.linalg.norm(H - sciH))
    print("SparseNMF error on H:", np.linalg.norm(H - spH))
    print()
    print("Scipy error on W:", np.linalg.norm(W - sciW))
    print("SparseNMF error on W:", np.linalg.norm(W - spW))
    print()
    print("Scipy error on X:", np.linalg.norm(X - np.dot(sciW, sciH)))
    print("SparseNMF error on X:", np.linalg.norm(X - np.dot(spW, spH)))






                









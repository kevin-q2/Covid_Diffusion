import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.sparse import random




#############################################################################################################

# An algorithm for diffusion based matrix factorization

# Uses the original ideas from Non Negative Matrix Factorization:
# https://papers.nips.cc/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf
# 
# And NMF with sparseness constraints
# https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf

# But contributes the idea of adding another factor to the equation
# So that an input n x m matrix D can be factorized into parts:
# X : A n x k matrix of basis vectors (colunwise)
# V: A sparse k x m matrix which represents an initial state for a
#    diffusion process that happens in K
# K: an input m x m matrix that represents some process of diffusion in an m x m graph

# INPUT:
# D : The original n x m data matrix
# K : m x m diffusion matrix
# ncomponents: desired number of basis vectors
# sparseness : desired level of sparseness for V
#             (1 being totally sparse and 0 being a totally full matrix)
# iterations : max number of iterations to run through before landing on a solution

# OUTPUT:
# X : n x k basis vector matrix
# V : k x m sparse factor matrix


#############################################################################################################





class DiffusionNMF:
    def __init__(self, D, K, ncomponents, sparseV = None, sparseX = None, iterations = 500, tol = 1e-10):
        if np.any(np.array(D) < 0):
            raise ValueError('Input array is negative')

        self.D = np.array(D)
        self.K = np.array(K)
        self.ncomponents = ncomponents
        self.sparseV = sparseV
        self.sparseX = sparseX
        self.iterations = iterations
        self.tol = tol
    


    def projection(self, s, k1, k2):
        # This method projects a vector s into its closest non-negative counterpart
        # More specifically: it "finds a vector v having sum(abs(v)) = k1 (L1 norm)
        # and sum (v^2) = k2 (L2 norm) which is closest to s in the euclidean sense...restricted to being non-negative"
        #
        # This is the main muscle for making sure that V is sparse in the output

        # More details can be found within the original paper (link above)

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

            w = v - midpoints

            a = sum(w**2)
            b = 2*np.dot(w,v)  
            c = sum(v**2) - k2

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

        vdim = self.D.shape[0]
        samples = self.D.shape[1]

        # Initialize W and H as random matrices
        self.X = np.random.rand(len(self.D), self.ncomponents)
        self.V = np.random.rand(self.ncomponents, len(self.D[0]))

        # Normalize the L2 norm for each row of H
        self.V = normalize(self.V, norm = 'l2', axis = 0) # NOT SURE IF I CAN just use the scipy normalize or not ...

        # Adjust for sparseness 
        # (based on the sparseness equation in the paper)

        if self.sparseV is not None:
            L1s = math.sqrt(samples) - (math.sqrt(samples) - 1)*self.sparseV
            for i in range(self.V.shape[0]):
                self.V[i,:] = self.projection(self.V[i,:], L1s, 1)

        if self.sparseX is not None:
            L1a = math.sqrt(vdim) - (math.sqrt(vdim) - 1)*self.sparseX
            for i in range(self.X.shape[1]):
                self.X[:,i] = self.projection(self.X[:,i], L1a, 1)


        # initial objective (cost function)
        #O = 0.5 * ((self.X - np.dot(self.W,self.H)) ** 2).sum()
        O = np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K)))
 
        # initial step sizes
        x_step = 1
        v_step = 1
        v_done = False
        x_done = False
        iteri = self.iterations
        while iteri > 0 and not (v_done and x_done):
            # save previous values
            x_old = self.X 
            v_old = self.V


            # Update V    
            if self.sparseV is not None and v_done == False:
                grad1 = np.dot(self.X.T, np.dot(self.D, self.K.T))
                grad2 = np.dot(self.X.T, np.dot(self.X, np.dot(self.V, np.dot(self.K, self.K.T))))
                dV = grad1 - grad2
  
                # gradient descent until we've taken a sufficient step
                while True:
                    # gradient
                    v_new = self.V + v_step * dV
                    # project to the l1 norm determined by sparseness
                    for r in range(self.V.shape[0]):
                        v_new[r,:] = self.projection(v_new[r,:], L1s, 1)

                    # check if we've improved the distance
                    #dist = 0.5 * ((self.X - np.dot(self.W,h_new)) ** 2).sum()
                    dist = np.linalg.norm(self.D - np.dot(self.X, np.dot(v_new,self.K)))

                    if dist <= O:
                        # if we have then break
                        O = dist
                        v_step = v_step * 1.2
                        self.V = v_new
                        break

                    # else decrease the stepsize and try again
                    else:
                        v_step = v_step / 2

                    # If the stepsize is small enough then we've converged on a solution
                    
                    if v_step < 1e-200:
                        print("V Algorithm Converged")
                        v_done = True
                        break
                        #return
                    


            elif v_done == False:
                # standard NMF update step
                # because H is not constrained
                num = np.multiply(self.V, np.dot(self.X.T, np.dot(self.D, self.K.T)))
                denom = np.dot(self.X.T, np.dot(self.X, np.dot(self.V, np.dot(self.K, self.K.T)))) + 1e-9 # add 1e-9 to make sure its not 0?
                self.V = np.divide(num, denom)

                # re - normalize
                norms = (sum(np.transpose(self.V) ** 2)) ** 0.5 # l2 norm along rows of H

                # rows of H keep constant energy 
                # W is scaled up to accomodate
                for n in range(len(norms)):
                    self.V[n,:] /= norms[n]
                    self.X[:,n] *= norms[n]

                #O = 0.5 * ((self.X - np.dot(self.W,self.H)) ** 2).sum()
                change = O - np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K)))
                O = np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K)))
                if change < self.tol:
                    return



            # Update X
            # sparse nmf update step
            if self.sparseX is not None and x_done == False:
                # gradient step
                dX = np.dot(self.D, np.dot(self.K.T, self.V.T)) - np.dot(self.X, np.dot(self.V, np.dot(self.K, np.dot(self.K.T, self.V.T))))

                # loop until we decrease the objective
                while True:
                    x_new = self.X + x_step*dX
                    l2 = (sum(x_new ** 2)) ** 0.5 # columnwise l2 norm

                    # project to l1 sparseness and keep original l2 
                    for c in range(self.ncomponents):
                        x_new[:,c] = self.projection(x_new[:,c], L1a*l2[c], l2[c]**2)

                    # check if we've improved the distance
                    #dist = 0.5 * ((self.X - np.dot(w_new,self.H)) ** 2).sum()
                    dist = np.linalg.norm(self.D - np.dot(x_new, np.dot(self.V, self.K)))

                    if dist <= O:
                        self.X = x_new
                        O = dist
                        x_step = x_step * 1.2
                        break

                    else:
                        x_step /= 2

                    
                    if x_step < 1e-200:
                        print("X Algorithm Converged")
                        x_done = True
                        break

            # standard NMF update step
            # because W is not constrained
            elif x_done == False:
                num = np.dot(self.D, np.dot(self.K.T, self.V.T))
                denom = np.dot(self.X, np.dot(self.V, np.dot(self.K, np.dot(self.K.T, self.V.T)))) + 1e-9
                lr = np.divide(num, denom)
                self.X = np.multiply(self.X, lr)

                #O = 0.5 * ((self.X - np.dot(self.W,self.H)) ** 2).sum()
                change = O - np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K)))
                O = np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K)))
                if change < self.tol:
                    return

            iteri -= 1



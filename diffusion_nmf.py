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
    def __init__(self, D, K, ncomponents, sparseV = None, sparseX = None, iterations = 500, tol = 1e-10, x_init = None, v_init = None, proj = False):
        if np.any(np.array(D) < 0):
            raise ValueError('Input array is negative')

        self.D = np.array(D)
        self.K = np.array(K)
        self.ncomponents = ncomponents
        self.sparseV = sparseV
        self.sparseX = sparseX
        self.iterations = iterations
        self.tol = tol
        self.x_init = x_init
        self.v_init = v_init
        self.proj = proj
    


    def projection(self, s, k1, k2):
        # This method projects a vector s into its closest non-negative counterpart
        # More specifically: it "finds a vector v having sum(abs(v)) = k1 (L1 norm)
        # and sum (v^2) = k2 (L2 norm) which is closest to s in the euclidean sense...restricted to being non-negative"
        #
        # This is the main muscle for making sure that V is sparse in the output

        # More details can be found within the original paper (link above)
        s = np.array(s)
        N = len(s)

        # project to the sum constraint:
        v = s + (k1 - sum(s))/N
        #print(s)
        #print(v)

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

            '''
            a = sum(w**2)
            b = 2*np.dot(w,v)  
            c = sum(v**2) - k2
            '''

            a = sum(w**2)
            b = 2*np.dot(midpoints, w)
            c = sum(midpoints**2)  - k2
            

            roots  = np.roots([a,b,c])

            try:
                if np.isreal(roots[0]) and np.isreal(roots[1]):
                    alpha = max(roots)
                elif np.isreal(roots[0]):
                    alpha = roots[0]
                elif np.isreal(roots[1]):
                    alpha = roots[1]
                else:
                    alpha = max(np.real(roots))


                # project to closest point on the joint constraint hypershpere
                #v = alpha*w + v
                v = midpoints + alpha * w

            except IndexError:
                # if no roots:
                # normalize based on the L2 constraint manually
                print("weird error")
                print("V:")
                print(v)
                print("Midpoints")
                print(midpoints)

                #if np.linalg.norm(v) > k2:
                #    v = (v / np.linalg.norm(v)) * k2
                v = midpoints

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

        # initialize X and V
        if self.x_init is None:
             self.X = np.random.rand(len(self.D), self.ncomponents)
        else:
            self.X = self.x_init

        if self.v_init is None:
            self.V = np.random.rand(self.ncomponents, len(self.D[0]))
        else:
            self.V = self.v_init

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
        O = np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K)))
 
        # initial step sizes
        x_step = 1
        v_step = 1

        iteri = self.iterations
        while iteri > 0:
            # save previous values
            x_old = self.X 
            v_old = self.V
            old_dist = O


            # Update V    
            if self.sparseV is not None:
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

                    # decrease step size until distance decreases or we've converged on a solution
                    else:
                        v_step /= 2
                        if v_step < 1e-100:
                            #print("V Algorithm Converged")
                            #v_done = True
                            break
                            #return

                    


            else:
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

                # project to closest sparse counterpoint
                if self.proj == True:
                    for r in range(self.V.shape[0]):
                        self.V[r,:] = self.projection(self.V[r,:], L1s, 1)

                O = np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K)))



            # Update X
            # sparse nmf update step
            if self.sparseX is not None: # and x_done == False:
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

                        if x_step < 1e-100:
                            break


            # standard NMF update step
            # because W is not constrained
            else: #elif x_done == False:
                num = np.dot(self.D, np.dot(self.K.T, self.V.T))
                denom = np.dot(self.X, np.dot(self.V, np.dot(self.K, np.dot(self.K.T, self.V.T)))) + 1e-9
                lr = np.divide(num, denom)
                self.X = np.multiply(self.X, lr)

                change = abs(O - np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K))))
                if change < self.tol:
                    x_done = True

                O = np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K)))


            # calculate change in cost and return if its small enough
            O =  np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V,self.K)))
            change = abs(old_dist - O)
            if change < self.tol:
                return

            iteri -= 1



    
    def least_square_solver(self, beta, eta):
        # solves the factorization with an alternating least squares solution

        # initialize X and V
        if self.x_init is None:
             self.X = np.random.rand(len(self.D), self.ncomponents)
        else:
            self.X = self.x_init



        iteri = self.iterations
        while iteri > 0:

            # Solve for V

            # stack matrices for sparsity constraint
            e = [[math.sqrt(beta)]*self.ncomponents]
            x_stack = np.append(self.X, e, axis = 0)

            zer_array = [[0]*self.D.shape[1]]
            d_stack = np.append(self.D, zer_array, axis = 0)

            # Least squares solution
            self.V = np.dot(np.linalg.lstsq(x_stack, d_stack)[0], np.linalg.inv(self.K))

            # Solve for X
            I = np.identity(self.ncomponents) * math.sqrt(eta)
            kht_stack = np.append(np.dot(self.K.T, self.V.T), I, axis = 0)

            zer_mat = np.zeros((self.ncomponents, self.D.shape[0]))
            dt_stack = np.append(self.D.T, zer_mat, axis = 0)

            # least squares solution:
            self.X = np.linalg.lstsq(kht_stack, dt_stack)[0].T

            iteri -= 1



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
# M : Mask for known values (matrix with 0's corresponding to unknowns and 1's for knowns)
# sparseness : desired level of sparseness for V
#             (1 being totally sparse and 0 being a totally full matrix)
# iterations : max number of iterations to run through before landing on a solution

# OUTPUT:
# X : n x k basis vector matrix
# V : k x m sparse factor matrix


#############################################################################################################





class DiffusionNMF:
    def __init__(self, D, K, ncomponents, M = None, iterations = 500, tol = 1e-10, x_init = None, v_init = None):
        if np.any(np.array(D) < 0):
            raise ValueError('Input array is negative')

        self.D = np.array(D)
        self.K = np.array(K)
        self.ncomponents = ncomponents
        self.masking = True
        if M is None:
            self.M = np.ones(np.shape(D))
            self.masking = False
        else:
            self.M = np.array(M)
        self.iterations = iterations
        self.tol = tol
        self.x_init = x_init
        self.v_init = v_init
    
        

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
                # if no roots (v == midpoints):
                # normalize based on the L2 constraint manually
                #print("weird error")

                if np.linalg.norm(v) > k2:
                    v = (v / np.linalg.norm(v)) * k2
                #v = midpoints

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



    def MultUpdate(self):
        # Update based on standard multiplicative update rules

        # Adjust for Masked Values:
        #masked_d = self.D
        #masked_xvk = np.dot(self.X, np.dot(self.V, self.K))
        #if self.masking:

        masked_d = np.multiply(self.M, self.D)
        masked_xvk = np.multiply(self.M, (np.dot(self.X, np.dot(self.V, self.K))))

        # Update X
        num_x = np.dot(masked_d, np.dot(self.K.T, self.V.T))
        denom_x = np.dot(masked_xvk, np.dot(self.K.T, self.V.T)) + 1e-9
        #num_x = np.dot(self.D, np.dot(self.K.T, self.V.T))
        #denom_x = np.dot(self.X, np.dot(self.V, np.dot(self.K, np.dot(self.K.T, self.V.T)))) + 1e-9
        lr = np.divide(num_x, denom_x)
        self.X = np.multiply(self.X, lr)
        masked_xvk = np.multiply(self.M, (np.dot(self.X, np.dot(self.V, self.K))))


        # Update V
        num_v = np.dot(self.X.T, np.dot(masked_d, self.K.T))
        denom_v = np.dot(self.X.T, np.dot(masked_xvk, self.K.T)) + 1e-9 # add 1e-9 to make sure its not 0?
        #num_v = np.dot(self.X.T, np.dot(self.D, self.K.T))
        #denom_v = np.dot(self.X.T, np.dot(self.X, np.dot(self.V, np.dot(self.K, self.K.T)))) + 1e-9
        lrv = np.divide(num_v, denom_v)
        self.V = np.multiply(self.V, lrv)


    def HoyerProjection(self, sparseness):
        # Update with a modified version of Hoyer's NMF with sparseness constraints 
        #   https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf

        # Adjust for Masked Values:
        masked_d = self.D
        masked_xvk = np.dot(self.X, np.dot(self.V, self.K))
        if self.masking:
            masked_d *= self.M
            masked_xvk *= self.M

        # Update X
        # standard multiplicative update (Dont need X to be sparse)
        num_x = np.dot(masked_d, np.dot(self.K.T, self.V.T))
        denom_x = np.dot(masked_xvk, np.dot(self.K.T, self.V.T)) + 1e-9
        lr = np.divide(num_x, denom_x)
        self.X = np.multiply(self.X, lr)

        # Update V
        # projected sparseness update
        grad1 = np.dot(self.X.T, np.dot(masked_d, self.K.T))
        grad2 = np.dot(self.X.T, np.dot(masked_xvk, self.K.T))
        dV = grad1 - grad2

        # gradient descent until we've taken a sufficient step
        while True:
            # gradient
            v_new = self.V + self.v_step * dV

            # project to the l1 norm determined by sparseness
            for r in range(self.V.shape[0]):
                v_new[r,:] = self.projection(v_new[r,:], sparseness, 1)

            # check if we've improved the distance
            curr_dist = np.linalg.norm(self.M * (self.D - np.dot(self.X, np.dot(self.V,self.K))))
            new_dist = np.linalg.norm(self.M * self.D - np.dot(self.X, np.dot(v_new,self.K)))

            if new_dist < curr_dist:
                # if we have then break
                self.v_step *= 1.2
                self.V = v_new
                break

            # decrease step size until distance decreases or we've converged on a solution
            else:
                self.v_step /= 2
                if self.v_step < 1e-100:
                    break

    
    def MultProjection(self, sparseness):
        # a modified version of standard multiplicative update that
        # uses Hoyer's projection algorithm after the update step

        self.MultUpdate()
        for r in range(self.V.shape[0]):
            self.V[r,:] = self.projection(self.V[r,:], sparseness, 1)



    def HoyerSparse(self, lambda_sparse):
        self.X = normalize(self.X, axis = 0)
        # Update X by modifying the step size until cost decreases
        dX = np.dot(self.D, np.dot(self.K.T, self.V.T)) - np.dot(self.X, np.dot(self.V, np.dot(self.K, np.dot(self.K.T, self.V.T))))

        # loop until we decrease the objective
        while True:
            x_new = self.X + self.x_step*dX
            
            # set negative values to 0
            # and normalize columns of x


            # check if we've improved the distance
            #dist = 0.5 * ((self.X - np.dot(w_new,self.H)) ** 2).sum()
            curr_dist = np.linalg.norm(self.D - np.dot(self.X, np.dot(self.V, self.K)))
            new_dist = np.linalg.norm(self.D - np.dot(x_new, np.dot(self.V, self.K)))

            if new_dist < curr_dist:
                self.X = x_new
                self.x_step = self.x_step * 1.2
                break

            else:
                self.x_step /= 2

                if self.x_step < 1e-100:
                    break
        
        self.X[self.X < 0] = 0
        self.X = normalize(self.X, axis = 0)
        
        # Update V
        # Multiplicative update step modified to include sparseness
        # because H is not constrained
        num = np.multiply(self.V, np.dot(self.X.T, np.dot(self.D, self.K.T)))
        denom = np.dot(self.X.T, np.dot(self.X, np.dot(self.V, np.dot(self.K, self.K.T)))) + lambda_sparse + 1e-9 # add 1e-9 to make sure its not 0?
        self.V = np.divide(num, denom)
        #self.V = normalize(self.V)



    def least_squares(self, beta, eta):
        # Solve for X
        I = np.identity(self.ncomponents) * math.sqrt(eta)
        kht_stack = np.append(np.dot(self.K.T, self.V.T), I, axis = 0)

        zer_mat = np.zeros((self.ncomponents, self.D.shape[0]))
        dt_stack = np.append(self.D.T, zer_mat, axis = 0)

        # least squares solution:
        self.X = np.linalg.lstsq(kht_stack, dt_stack)[0].T
        # remove all negative values
        self.X[self.X < 0] = 0


        # Solve for V

        # stack matrices for sparsity constraint
        e = [[math.sqrt(beta)]*self.ncomponents]
        x_stack = np.append(self.X, e, axis = 0)

        zer_array = [[0]*self.D.shape[1]]
        d_stack = np.append(self.D, zer_array, axis = 0)

        # Least squares solution
        self.V = np.dot(np.linalg.lstsq(x_stack, d_stack)[0], np.linalg.inv(self.K))
        # remove all negative values
        self.V[self.V<0] = 0



    def solver(self, algorithm = 'MultUpdate', sparseness = None, lambda_v = None, beta = None, eta = None):
        # solves the Diffusion NMF problem with various different algorithms

        # INPUT: 

            # 'MultUpdate' -- default, standard multiplicative update solver (No sparseness)

            # 'HoyerP' -- Hoyer algorithm with projection to sparseness
                # requires sparseness parameter (value between 0 and 1 -- 1 being totally sparse)

            # 'MultProj' -- multiplicative update modified with projection to sparseness
                # requires sparseness parameter

            # 'HoyerS' -- Original Hoyer sparse coding algorithm (NO projection) 
                # requires lambda_v parameter (weight given to cost of V's L1 norm) -- must be >= 0

            # 'TwoPhase' -- two phase algorithm that uses MultUPdate and HoyerP
                # requires sparseness parameter

            # 'LeastSquares' -- modified version of standard alternating least squares algorithm
                # requires both a beta and eta parameter (weights)

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


        # Some sparseness constraints to use depending on the algorithm
        if sparseness is not None:
            # Normalize the L2 norm for each row of H
            self.V = normalize(self.V, norm = 'l2', axis = 1) # NOT SURE IF I CAN just use the scipy normalize or not ...

            L1s = math.sqrt(samples) - (math.sqrt(samples) - 1)*sparseness
            for i in range(self.V.shape[0]):
                self.V[i,:] = self.projection(self.V[i,:], L1s, 1)
        


        # initial objective (cost function)
        O = np.linalg.norm(self.M * (self.D - np.dot(self.X, np.dot(self.V,self.K))))

        # step sizes for iterating in projected gradient descent
        self.x_step = 1
        self.v_step = 1
        iteri = self.iterations

        while iteri > 0:
            x_old = self.X 
            v_old = self.V
            old_dist = O

            # These algorithms will update self.X and self.V every iteration
            if algorithm == 'HoyerP':
                self.HoyerProjection(L1s)
            elif algorithm == 'MultProj':
                self.MultProjection(L1s)
            elif algorithm == 'HoyerS':
                self.HoyerSparse(lambda_v)
            elif algorithm == 'TwoPhase':
                if iteri > self.iterations / 2:
                    self.MultUpdate()
                else:
                    self.HoyerProjection(L1s)
            elif algorithm == 'LeastSquares':
                self.least_squares(beta, eta)
            else:
                # standard multiplicative update
                self.MultUpdate()

            #self.V = normalize(self.V, norm = 'l1', axis = 1)
            # calculate change in cost and return if its small enough
            O =  np.linalg.norm(self.M * (self.D - np.dot(self.X, np.dot(self.V,self.K))))
            change = abs(old_dist - O)
            if change < self.tol:
                return

            iteri -= 1


        




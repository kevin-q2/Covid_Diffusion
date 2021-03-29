import numpy as np
import pandas as pd
import random
from copy import copy, deepcopy

###############################################################################################################

# This is an attempt at implementing the kmeans-- algorithm 
#
# which is outlined here: http://users.ics.aalto.fi/gionis/kmmm.pdf
#
# INPUT
# X: 2d array of data to cluster --- [[n dimensional point to cluster], [n dimensional point to cluster], ...]
# k: desired number of clusters
# l: number of outliers you expect
# max_iter (optional): number of iterations
#
# OUTPUT from .cluster()

# clusters: a dictionary with structure -- {cluster # : [0, 2, ...]}
#           where 0,2, etc. are the indexes of n-dimensional points in X  
#           ex) {0: [1], 1:[0]} -- means X[1] is in cluster 0 and X[0] is in cluster 1
#
# outliers: a list of points that are considered as outliers and have been removed from the clusters dict
#            ex) [1,3] -- means X[1] and X[3] are outliers

###############################################################################################################



class kmeans_minus_minus():
    def __init__(self, X, k, l, max_iter=500):
        self.X = X
        self.k = k
        self.l = l
        self.max_iter = 500
        

    def cluster(self):                                                                                                                                                          
        # start by randomly generating a length k-set of d dimensional points
        mini = np.amin(self.X)
        maxi = np.amax(self.X)

        C = [random.sample(np.linspace(mini,maxi, num=len(self.X[0])*2).tolist(), len(self.X[0])) for i in range(self.k)]
        M = list(range(len(self.X)))

        for i in range(self.max_iter):
            clusters = {g:[] for g in range(self.k)} 

            # for each point calculate the distance to its nearest cluster center
            d = {r:np.nan for r in M}
            for j in M:
                ds = -1
                clus = -1
                for c in range(len(C)):
                    mop = np.linalg.norm(np.array(self.X[j]) - np.array(C[c]))
                    if mop < ds or ds == -1:
                        ds = mop
                        clus = c
                d[j] = ds
                clusters[clus].append(j)
        
            # Reorder the data points in X by decreasing order of 
            # distance to their nearest center
            M = sorted(d, key=d.get)
            M.reverse()

            # take out l outliers before updating the centers:
            L = M[:self.l]
            Li = copy(L)
            Mi = M[self.l:]

            # Update the centers:
            for h in range(self.k):
                for out in Li:
                    if out in clusters[h]:
                        clusters[h].remove(out)
                        Li.remove(out)

                P = np.array([self.X[col] for col in clusters[h]])
                new_center = P.mean(axis=0)
                C[h] = new_center

        return clusters, L
        








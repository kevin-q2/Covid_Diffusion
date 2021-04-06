import numpy as np
import pandas as pd
import random
from sklearn.cluster import _kmeans
from copy import copy, deepcopy

###############################################################################################################

# This is an attempt at implementing the kmeans algorithm 
#
# which is outlined here: http://users.ics.aalto.fi/gionis/kmmm.pdf
#
# INPUT
# X: 2d array of data to cluster --- [[n dimensional point to cluster], [n dimensional point to cluster], ...]
# k: desired number of clusters
# l: number of outliers you expect
# max_iter (optional): number of iterations
# tol (optional): minimum change in cluster centers before declaring convergence 
#
# OUTPUT from .cluster()
#
# best_labels: A list of cluster labels for the original data
#              ex) [0, 1, 0] would mean that X[0] and X[2] are in cluster 0 and X[1] is in cluster 1
#
#       Note: a label of -1 means the point is an outlier
#              ex) [0, 1, 0, -1] would mean that X[3] is an outlier and has not been placed in a cluster
# 

###############################################################################################################



class kmeans_minus_minus():
    def __init__(self, X, k, l, max_iter=1000, tol = 1e-10):
        self.X = np.array(X)
        self.k = k
        self.l = l
        self.max_iter = max_iter
        self.tol = tol
 

    def cluster(self):       

        # Use sklearn to get a kmeans++ initialization of cluster centers
        try:
            # for older version of sklearn
            C = _kmeans._init_centroids(self.X, self.k, "k-means++")
        except:
            # Updated sklearn
            C, oth = _kmeans.kmeans_plusplus(self.X, self.k)

        best_labels = None
        best_inertia = None
        best_centers = C

        for i in range(self.max_iter):
            labels = np.array([np.nan for lab in range(len(self.X))])

            # for each point calculate the distance to its nearest cluster center
            # label each point with its closest cluster center
            d = {}
            for j in range(len(self.X)):
                ds = None
                clus = None
                for c in range(len(C)):
                    mop = np.linalg.norm(np.array(self.X[j]) - np.array(C[c]))
                    if ds is None or mop < ds:
                        ds = mop
                        clus = c
                d[j] = ds
                labels[j] = clus


            # Reorder the data points in X by decreasing order of 
            # distance to their nearest center
            km = sorted(d, key=d.get)
            km.reverse()

            # take out l outliers before calculating inertia / updating the centers:
            # (Kmeans minus minus step)
            L = km[:self.l]
            Mi = km[self.l:]
            for l in L:
                labels[l] = -1

            # Calculate inertia 
            inertia = 0
            for m in Mi:
                inertia += d[m] ** 2

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_inertia = inertia
                best_centers = C


            # Update the centers for next iteration:
            old = C.copy()
            for h in range(self.k):
                ind = np.where(labels == h)[0]
                P = np.array([self.X[col] for col in ind])
                new_center = P.mean(axis=0)
                C[h] = new_center

            # OR break if we've reached convergence:
            shift = np.linalg.norm(C - old)
            if shift <= self.tol:
                break


        return best_labels.astype(int)
    
        








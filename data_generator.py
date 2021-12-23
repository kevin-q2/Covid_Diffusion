import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

#################################################################################
# A simple function for generating random W,H matrices to be used 
# together with a diffusion kernel as input for our algorithm 
#
# INPUT:
# n - number of rows in the W matrix
# m - number of columns in H matrix
# rank - integer representing desired rank of the data, 
#           determines the shape of the output W,H
#
# state - integer for random seed used to help reproduce results
#
# OUTPUT: 
# W - a n x rank matrix of sin waves with differing frequency and 
#   amplitude with values strictly between 0 and 1
#
# H - a rank x m sparse matrix with entries between 0 and 1
#
# Note: the matrix product of W and H produces an n x m matrix
#
#################################################################################


def gen_decomposition(n, m, rank, state = None):
    
    # Generate random H with scipy's sparse random
    test_h = sp.random(rank, m, density = 0.07).A
    
    # generate new H if we have any zero rows
    while np.where(np.sum(test_h, axis = 1) == 0)[0].size != 0:
        test_h = sp.random(rank, m, density = 0.07, random_state = state).A
     
    # Normalize row sums of H so that they all sum to 1
    for g in range(len(test_h)):
        scal = test_h[g,:].sum()
        test_h[g,:] /= scal

    H = pd.DataFrame(test_h)
    
    
    # generate sin waves of data for W
    time = np.linspace(1,n,n)
    np.random.seed(state)
    freqs = np.random.normal(0,0.2,rank)
    waves = np.outer(time, freqs)
    for col in range(waves.shape[1]):
        waves[:,col] = (col + 1) * np.sin(waves[:,col])
    
    # Normalize W to lie between 0 and 1
    W = pd.DataFrame(waves)
    W /= W.max().max() * 2
    W += 0.5
    
    return W,H

########################################################################
# Another quick function to generate laplacian matrices
# Simply generates a random erdos renyi graph based on given
# inputs:
#
# size - number of nodes in the graph
# p_edge - probability that an edge exist between any two nodes
# state - random seed for graph generation
#
# And outputs the corresponding laplacian matrix of the graph
########################################################################

def gen_laplacian(size, p_edge = 0.1, state = None):
    graph = nx.generators.random_graphs.erdos_renyi_graph(n = size, p = 0.1, seed = state)
    laplacian = nx.linalg.laplacianmatrix.laplacian_matrix(graph)
    return laplacian.toarray()
import os 
import pandas as pd
import numpy as np
import math
import heapq as hq #min heap for dijkstra's
from adj_list import *
from state_adj import make_graph


###################################################################################

# This will be where I have all graph related functions 
# The graph in question is laid out in adj_list.py
# and created for states/counties in state_adj.py and county_adj.py

###################################################################################







####################################################################################

# Graph initialization functions:

def zero_one(graph, wave, to_decide):
    # takes a graph and for every node it sets the val attribute 
    # to 0 or 1 based on the values passed by to_decide and wave

    # For example: If considering wave 1 and to_decide = 0.5
    # for each node, if its basis 1 factor is above 0.5 change val to 1
    # else leave it at 0

    for node in graph.V.keys():
        if graph.V[node].surge_vals[wave] >= to_decide:
            graph.V[node].val = 1

#####################################################################################






########################################################################################

# Shortest path algorithms:

def dijkstra_adj(graph, source, weighted = True):
    # FOR a homemade ADJ list
    # Input: a graph made from adj_list.py and the name of a source node
    #       + a variable weighted which should be set to true if using a weighted graph
    #
    # Output: two dictionaries of size N, in which each key represents a node n:
    #               distances: the shortest distance from s to n
    #               parents: the last (previous) node before n on the shortest path

    distances = {i:None for i in graph.V.keys()}
    parents = {i:None for i in graph.V.keys()}

    pi = {i:math.inf for i in graph.V.keys()} # list of tentative shortest paths
    pi[source] = 0

    h = []
    hq.heappush(h, (pi[source], source)) # min heap used as priority queue
    while(len(h) > 0):
        ext = hq.heappop(h)
        if(distances[ext[1]] is None):
            distances[ext[1]] = ext[0]

            for v in graph.V[ext[1]].neighbors.keys():
                if weighted == True:
                    edge_length = graph.V[ext[1]].neighbors[v]
                else:
                    edge_length = 1

                if pi[v.name] > pi[ext[1]] + edge_length:
                    pi[v.name] = pi[ext[1]] + edge_length
                    parents[v.name] = ext[1]
                    hq.heappush(h, (pi[v.name], v.name))

    return distances, parents

def dijkstra_pd(frame, source):
    # works on a PANDAS dataframe

    # Input: an n x n pandas dataframe and the name of a source node
    #       + a variable weighted which should be set to true if using a weighted graph
    #
    # Output: two dictionaries of size N, in which each key represents a node n:
    #               distances: the shortest distance from s to n
    #               parents: the last (previous) node before n on the shortest path

    distances = {i:None for i in frame.columns}
    parents = {i:None for i in frame.columns}

    pi = {i:math.inf for i in frame.columns} # list of tentative shortest paths
    pi[source] = 0

    h = []
    hq.heappush(h, (pi[source], source)) # min heap used as priority queue
    while(len(h) > 0):
        ext = hq.heappop(h)
        if(distances[ext[1]] is None):
            # if it gets popped from the heap then I know that is the shortest distance (unless its been popped before)
            distances[ext[1]] = ext[0]

            # update shortest distance (so far) for all neighbors
            neighbors = frame.loc[:,ext[1]]
            neighbors = neighbors.loc[neighbors != 0].index
            for v in neighbors:
                edge_length = frame.loc[v,ext[1]]

                if pi[v] > pi[ext[1]] + edge_length:
                    pi[v] = pi[ext[1]] + edge_length
                    parents[v] = ext[1]
                    hq.heappush(h, (pi[v], v))

    return distances, parents

def dijkstra(graph, source, weighted):
    # handles the case of either data frame or adj list
    if type(graph) == pd.core.frame.DataFrame:
        return dijkstra_pd(graph, source)
    else:
        return dijkstra_adj(graph, source, weighted)


    
def k_shortest(graph, source, weighted = True, k = 1):
    # finds the  first k shortest paths in a graph
    # k = 1 would be the same as just running dijkstras

    distances = {i:[] for i in graph.V.keys()}
    parents = {i:[] for i in graph.V.keys()}

    # NEED to make a copy of the graph
    while k > 0:
        dist, par = dijkstra(graph, source, weighted)

        for path_node in par.keys():
            # remove the path from the adjacency dicts
            if par[path_node] is not None:
                to_pop = graph.V[par[path_node]].find_neighbor(path_node)
                graph.V[par[path_node]].neighbors.pop(to_pop)

        for reg in dist.keys():
            # append to the final results
            distances[reg].append(dist[reg])
            parents[reg].append(par[reg])

        k =  k - 1

    return distances, parents


    

#################################################################################################



#################################################################################################

# TESTING STUFF

if __name__ == '__main__':
    state_g = make_graph()
    
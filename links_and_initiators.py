import pandas as pd
import numpy as np
import math
import random
from graph_operation import *




def gn_given_m(m,g,n):
    # to find P(G,N | M):
    # P(G,N | M) = P(M | G, N) * P(G) * P(N)
    
    # P(G) * P(N)
    p_gn = (math.exp(-g.values.sum()/(n.values.sum() * 2))) * (math.exp(-n.values.sum()/(n.values.sum() * 2)))
    
    
    # FOR INFLUENCE based on current G:
    dists = pd.DataFrame(columns = m.columns, index = m.columns)
    for colz in dists.columns:
        dist_g = dijkstra(g,colz,weighted = False)[0]
        for con in dist_g.keys():
            dists.loc[con,colz] = dist_g[con]
            
    influence = 0.9 ** dists
    influence = influence.replace(np.nan, 0)


    # P(M | G, N):
    mgn = 1
    for cx in m.columns:
        for idx in m.index:
            
            # P(M(i,u)=0 | G,N)
            m_gn = 1 - n.loc[idx, cx]
            for ncx in n.columns:
                if ncx != cx:
                    m_gn *= (1 - n.loc[idx,ncx]*influence.loc[ncx, cx]) #state_influence(g,stx,ux)

            # P(M(i,u)=1 | G,N)
            if m.loc[idx,cx] == 1:
                m_gn = 1 - m_gn
            
            mgn *= m_gn

    return (p_gn * mgn)


def metropolis_hastings(m, g_init, n_init, steps):
    # The metropolis hastings algorithm implemented in the paper
    prev_prob = 1
    while steps > 0:
        gg = g_init.copy(deep = True)
        nn = n_init.copy(deep = True)
        # random local move:
        gorn = random.choice([0,1])
        ind = None
        col = None
        if gorn == 0:
            # change an element from G
            ind = random.choice(gg.index)
            col = random.choice(gg.columns)
            pos = gg.loc[ind,col]
            if pos == 0:
                #print("changing G -- 0 to 1")
                gg.loc[ind,col] = 1
            else:
                #print("changing G -- 1 to 0")
                gg.loc[ind,col] = 0
        else:
            # change an element from N
            ind = random.choice(nn.index)
            col = random.choice(nn.columns)
            pos = nn.loc[ind,col]
            if pos == 0:
                #print("changing N -- 0 to 1")
                nn.loc[ind,col] = 1
            else:
                #print("changing N -- 1 to 0")
                nn.loc[ind,col] = 0
                
        
        # probability of move:
        after_move = gn_given_m(m,gg,nn)
        diff = after_move / prev_prob

        p_move = min(1, diff)
        
        # make the move or decide not to
        decider = np.random.choice([0,1],p = [1-p_move, p_move])
        if decider == 1:
            #make the move
            g_init = gg
            n_init = nn
            # update previous probability
            prev_prob = after_move
            if gorn == 1:
                print("N change success")
            else:
                print("G change success")
        else:
            # Do nothing
            #print("change fail")
            pass
        
        steps -= 1
        
    return g_init, n_init




def average_runs(M,g0,n0,steps,runs):
    #returns the average G, N over multiple runs
    g_avg = pd.DataFrame(columns = g0.columns, index = g0.index)
    g_avg = g_avg.replace(np.nan, 0)
    n_avg = pd.DataFrame(columns = n0.columns, index = n0.index)
    n_avg = n_avg.replace(np.nan, 0)
    
    for run in range(runs):
        print("Run:", run)
        g,n = metropolis_hastings(M, g0, n0,steps)
        g_avg = g_avg.add(g)
        n_avg = n_avg.add(n)
        
    g_avg /= runs
    n_avg /= runs
    
    for col in g_avg.columns:
        for ind in g_avg.index:
            if g_avg.loc[ind,col] > 0.2:
                g_avg.loc[ind,col] = 1
            else:
                g_avg.loc[ind,col] = 0
                
    for col in n_avg.columns:
        for ind in n_avg.index:
            if n_avg.loc[ind,col] > 0.2:
                n_avg.loc[ind,col] = 1
            else:
                n_avg.loc[ind,col] = 0
    return g_avg, n_avg



def links_initiators(M, steps, runs):
    # Given a matrix M find a matrix of initiators N and graph of connections
    N0 = M
    G0 = pd.DataFrame(columns = M.columns, index = M.columns)
    G0 = G0.replace(np.nan, 0)
    G,N = average_runs(M,G0,N0,steps,runs)
    return G,N




# some stuff I'm not using right now:
'''
state_distance = pd.DataFrame(columns = cc.columns, index = cc.columns)
for colz in state_distance.columns:
    dist_g = dijkstra(state_graph,colz)[0]
    for con in dist_g.keys():
        state_distance.loc[con,colz] = dist_g[con]

def state_influence(G, i, j):
    # influence of node i upon node j in graph G
    # helps model diffusion?
    
    #dist_g = dijkstra(state_graph,i)[0]
    #dist = dist_g[j]
    
    #dist = state_distance.loc[i,j]
    dist = dijkstra(G, i, weighted = False)[0][j]
    
    if dist is None:
        return 0
    else:
        #spreader1 = nmfed.y_table.loc['basis 0', i]
        #spreader2 = nmfed.y_table.loc['basis 0', j]
        #coff = 1 / abs(spreader1- spreader2)
        return (0.90 ** (dist))
'''
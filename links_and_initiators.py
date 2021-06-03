import pandas as pd
import numpy as np
import math
import random
import os
import sys
import heapq as hq #min heap for dijkstra's
from matrix_operation import mat_opr


class links_initiators:
    def __init__(self, M, steps, samples, alpha, c1, c2):
        # The observed matrix you wish to describe
        self.M = M 

        # The number of steps to be used in Metropolis Hastings
        # Note that steps/2 steps will be automatically used as a burn in period so initialize accordingly
        self.steps = steps

        # The number of average G's and N's to return
        self.samples = samples
        
        # diffusion parameter (between 0 and 1):
        self.alpha = alpha
        
        #constants for determining P(G) and P(N) respectively
        self.c1 = c1
        self.c2 = c2


        # Returned lists of sampled Gs and Ns
        self.G = []
        self.N = []

        # Used in calculation of gn_given_m()
        infer = pd.DataFrame(columns = self.M.columns, index = self.M.columns)
        infer = infer.replace(np.nan, 0.0)
        self.influence = infer.astype(np.float64)

        # Some optional attributes
        self.G0 = None
        self.N0 = None




    def dijkstra_pd(self, frame, source):
        # Shortest path algorithm
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



    def average_runs(self, g_sum, n_sum, g_changes, n_changes):
        # helper function for metropolis hastings to find the average dataframe over a 
        # period of sampling
            
        g_avg = g_sum / g_changes
        n_avg = n_sum / n_changes
        
        for col in g_avg.columns:
            for ind in g_avg.index:
                if g_avg.loc[ind,col] > 0.5:
                    g_avg.loc[ind,col] = 1
                else:
                    g_avg.loc[ind,col] = 0
                    
        for col in n_avg.columns:
            for ind in n_avg.index:
                if n_avg.loc[ind,col] > 0.5:
                    n_avg.loc[ind,col] = 1
                else:
                    n_avg.loc[ind,col] = 0
                    
        return g_avg, n_avg




    def gn_given_m(self, g,n,g_move = True):
        # Helper for metropolis hastings
        # to find P(G,N | M):
        # P(G,N | M) = P(M | G, N) * P(G) * P(N)
        
        # P(G) * P(N)
        p_gn = (math.exp(-g.values.sum() * self.c1)) * (math.exp(-n.values.sum() * self.c2))
        
        
        # FOR INFLUENCE based on current G:
        if g_move == True or self.influence is None:
            dists = pd.DataFrame(columns = self.M.columns, index = self.M.columns)
            for colz in dists.columns:
                dist_g = self.dijkstra_pd(g,colz)[0]
                for con in dist_g.keys():
                    dists.loc[con,colz] = dist_g[con]
                    
            infl = 0.9 ** dists
            self.influence = infl.replace(np.nan, 0.0)

        '''
        if g_move == False and self.influence is not None:
            dists = pd.DataFrame(columns = self.M.columns, index = self.M.columns)
            for colz in dists.columns:
                dist_g = self.dijkstra_pd(g,colz)[0]
                for con in dist_g.keys():
                    dists.loc[con,colz] = dist_g[con]
                    
            infl = 0.9 ** dists
            infl = infl.replace(np.nan, 0.0)

            if not infl.equals(self.influence):
                print("Discrep")
        '''

        # P(M | G, N):
        mgn = 1
        for cx in self.M.columns:
            for idx in self.M.index:
                
                # P(M(i,u)=0 | G,N)
                m_gn = 1 - n.loc[idx, cx]
                for ncx in n.columns:
                    if ncx != cx:
                        m_gn *= (1 - n.loc[idx,ncx]*(self.influence.loc[ncx, cx])) #state_influence(g,stx,ux)

                # P(M(i,u)=1 | G,N)
                if self.M.loc[idx,cx] == 1:
                    m_gn = 1 - m_gn
                
                mgn *= m_gn

        return p_gn * mgn




    def metropolis_hastings(self, g_init, n_init, sample_steps, burner = False):
        # The metropolis hastings algorithm implemented in the paper

        # To collect and use for computing averages later:
        # sum of all g's and n's over the whole process
        g_changes = 1
        n_changes = 1
        g_sum = g_init.copy(deep = True)
        n_sum = n_init.copy(deep = True)

        prev_prob = np.float64
        prev_prob = self.gn_given_m(g_init,n_init, True)

        # walk over a given number of steps
        while sample_steps > 0:
            gg = g_init.copy(deep = True)
            nn = n_init.copy(deep = True)

            # random local move:
            g_or_n = random.choice([0,1])
            ind = None
            col = None
            if g_or_n == 0:
                # change an element from G
                ind = random.choice(gg.index)
                col = random.choice(gg.columns)
                pos = gg.loc[ind,col]
                if pos == 0:
                    #changing pos in G from 0 to 1
                    gg.loc[ind,col] = 1
                else:
                    #changing pos in G from 1 to 0
                    gg.loc[ind,col] = 0
            else:
                # change an element from N
                ind = random.choice(nn.index)
                col = random.choice(nn.columns)
                pos = nn.loc[ind,col]
                if pos == 0:
                    #changing pos in N from 0 to 1
                    nn.loc[ind,col] = 1
                else:
                    #changing pos N from 1 to 0
                    nn.loc[ind,col] = 0
                    
            
            # sampled probability of move:
            #after_move = np.float64()

            
            '''
            if g_or_n == 0:
                # run dijkstra for a change in G
                #print("G change")
                after_move = self.gn_given_m(gg,nn,True)
            else:
                # save time by skipping dijkstra (Only N was updated)
                #print("N change")
                after_move = self.gn_given_m(gg,nn, False)
            '''

            after_move = self.gn_given_m(gg,nn,True)
            #print("after move: ", after_move)
            #print("prev_prob: ", prev_prob)
            ratio = after_move / prev_prob
            #print()


            p_move = min(1, ratio)
            
            # make the move or decide not to
            decider = np.random.choice([0,1],p = [1-p_move, p_move])
            if decider == 1:
                #make the move
                g_init = gg
                n_init = nn
                
                if g_or_n == 0:
                    g_sum += g_init
                    g_changes += 1

                else:
                    n_sum += n_init
                    n_changes += 1

                
                # update previous probability
                prev_prob = after_move
            

            sample_steps -= 1

        if burner == True:
            return g_init, n_init
        else: 
            return self.average_runs(g_sum, n_sum, g_changes, n_changes)


    def find(self):
        # Given a matrix M find a matrix of initiators N and graph of connections
        #returns a list of average G, N over multiple runs
        # steps / 2 should be divisible by # of samples
        if self.N0 is None:
            N0 = self.M
        else:
            N0 = self.N0

        if self.G0 is None:
            G0 = pd.DataFrame(columns = self.M.columns, index = self.M.columns)
            G0 = G0.replace(np.nan, 0)
        else:
            G0 = self.G0

        #burn in period
        g_init, n_init = self.metropolis_hastings(G0,N0,self.steps/2,burner = True)
        print("After burn in: ")
        print(g_init)
        print(n_init)

        for run in range(self.samples):
            g_s,n_s = self.metropolis_hastings(g_init, n_init, (self.steps/2)/self.samples)
            self.G.append(g_s)
            self.N.append(n_s)
            g_init = g_s
            n_init = n_s

        return self.G,self.N








# Here I have modified the original object to work with some different kinds of input
# specifically for some M that isn't 0 or 1 for every entry
# Likewise the output is no longer 0 - 1 defined
class applied_links_initiators(links_initiators):
    def __init__(self, M, steps, samples, alpha, c1, c2):
        super().__init__(M, steps, samples, alpha, c1, c2)


    def gn_given_m(self, g,n,g_move = True):
        # Helper for metropolis hastings
        # to find P(G,N | M):
        # P(G,N | M) = P(M | G, N) * P(G) * P(N)
        
        # P(G) * P(N)
        p_gn = (math.exp(-g.values.sum() * self.c1)) * (math.exp(-n.values.sum() * self.c2))

        '''
        g_count = 0
        n_count = 0
        for g_i in g.values.flatten():
            if g_i > 0.7:
                g_count +=1

        for n_i in n.values.flatten():
            if n_i > 0.7:
                n_count +=1
        
        
        #p_gn = (math.exp(-g.values.sum() * self.c1)) * (math.exp(-n.values.sum() * self.c2))
        p_gn = (math.exp(-g_count * self.c1)) * (math.exp(-n_count * self.c2))
        '''
        
        
        # FOR INFLUENCE based on current G:
        if g_move == True or self.influence is None:
            dists = pd.DataFrame(columns = self.M.columns, index = self.M.columns)
            for colz in dists.columns:
                dist_g = self.dijkstra_pd(g,colz)[0]
                for con in dist_g.keys():
                    dists.loc[con,colz] = dist_g[con]
                    
            infl = 0.9 ** dists
            self.influence = infl.replace(np.nan, 0)


        # P(M | G, N):
        mgn = 1
        for cx in self.M.columns:
            for idx in self.M.index:
                
                # P(M(i,u)=0 | G,N)
                #m_gn = 1 - abs(self.M.loc[idx, cx] - n.loc[idx, cx])
                m_gn = 1 - n.loc[idx, cx]
                for ncx in n.columns:
                    if ncx != cx:
                        #m_gn *= (1 - abs(self.M.loc[idx, cx] - n.loc[idx,ncx]))*(self.influence.loc[ncx, cx]) #state_influence(g,stx,ux)
                        m_gn *= (1 - n.loc[idx,ncx]*self.influence.loc[ncx, cx]) #state_influence(g,stx,ux)
                
                # P(M(i,u)=1 | G,N)
                if self.M.loc[idx,cx] > self.M.mean().mean():
                    m_gn = 1 - m_gn

                mgn *= m_gn

        

        return p_gn * mgn

    def metropolis_hastings(self, g_init, n_init, sample_steps, burner = False):
        # The metropolis hastings algorithm implemented in the paper

        # To collect and use for computing averages later:
        # sum of all g's and n's over the whole process
        g_changes = 1
        n_changes = 1
        g_sum = g_init.copy(deep = True)
        n_sum = n_init.copy(deep = True)


        prev_prob = self.gn_given_m(g_init,n_init)

        # walk over a given number of steps
        while sample_steps > 0:
            gg = g_init.copy(deep = True)
            nn = n_init.copy(deep = True)

            # random local move:
            g_or_n = random.choice([0,1])
            ind = None
            col = None
            if g_or_n == 0:
                # change an element from G
                ind = random.choice(gg.index)
                col = random.choice(gg.columns)

            
                pos = gg.loc[ind,col]
                if pos== 0:
                    #changing pos in G from 0 to 1
                    gg.loc[ind,col] = 1
                else:
                    #changing pos in G from 1 to 0
                    gg.loc[ind,col] = 0
                
                
                '''
                #gg.loc[ind,col] = np.random.uniform(0,1)
                sample = abs(np.random.normal(0,0.3))
                if sample > 1:
                    sample = 1
                gg.loc[ind,col] = sample
                '''
            else:
                # change an element from N
                ind = random.choice(nn.index)
                col = random.choice(nn.columns)

                pos = nn.loc[ind,col]
                #nn.loc[ind,col] = random.choice([0,1])

                pos = nn.loc[ind,col]
                if pos== 0:
                    #changing pos in G from 0 to 1
                    nn.loc[ind,col] = 1
                else:
                    #changing pos in G from 1 to 0
                    nn.loc[ind,col] = 0
                '''
                # In this case sample from uniform distribution for new value
                #nn.loc[ind,col] = np.random.uniform(0,1)
                sample = abs(np.random.normal(0,0.3))
                if sample > 1:
                    sample = 1
                nn.loc[ind,col] = sample
                '''
                    
            
            # sampled probability of move:
            after_move = self.gn_given_m(gg,nn,True)
            ratio = after_move / prev_prob

            p_move = min(1, ratio)
            
            # make the move or decide not to
            decider = np.random.choice([0,1],p = [1-p_move, p_move])
            if decider == 1:
                #make the move
                g_init = gg
                n_init = nn
                
                if g_or_n == 0:
                    g_sum += g_init
                    g_changes += 1

                else:
                    n_sum += n_init
                    n_changes += 1
                
                # update previous probability
                prev_prob = after_move


            sample_steps -= 1

        if burner == True:
            return g_init, n_init
        else: 
            return self.average_runs(g_sum, n_sum, g_changes, n_changes)

    def find(self):
        # Given a matrix M find a matrix of initiators N and graph of connections
        #returns a list of average G, N over multiple runs
        # steps / 2 should be divisible by # of samples
        if self.N0 is None:
            N0 = self.M.copy(deep = True)
            for c in N0.columns:
                for i in N0.index:
                    if N0.loc[i,c] <= self.M.mean().mean():
                        N0.loc[i,c] = 0
                    else:
                        N0.loc[i,c] = 1
        else:
            N0 = self.N0

        if self.G0 is None:
            G0 = pd.DataFrame(columns = self.M.columns, index = self.M.columns)
            G0 = G0.replace(np.nan, 0)
        else:
            G0 = self.G0

        #burn in period
        g_init, n_init = self.metropolis_hastings(G0,N0,self.steps/2,burner = True)
        print("After burn in: ")
        print(g_init)
        print(n_init)

        for run in range(self.samples):
            g_s,n_s = self.metropolis_hastings(g_init, n_init, (self.steps/2)/self.samples)
            self.G.append(g_s)
            self.N.append(n_s)
            g_init = g_s
            n_init = n_s

        return self.G,self.N


if __name__ == '__main__':
    M_test2 = [[1,1,1,0,0,0],
          [1,1,1,1,1,1],
          [0,0,0,1,1,1]]
    M_test2 = pd.DataFrame(M_test2).T
    M_test2.columns = ['state 0', 'state 1', 'state 2']

    test_links = links_initiators(M_test2, 2000, 10, 0.9,2, 9)
    g,n = test_links.find()


'''
if __name__ == '__main__':
    cwd = os.getcwd()
    par = os.path.dirname(cwd)
    sys.path.append(par)

    state_dset = pd.read_csv('collected_data/state_dataset.csv', index_col = 0)
    state_dset = mat_opr(state_dset)
    population = pd.read_csv('collected_data/state_census_estimate.csv', index_col = 'NAME')

    state_iso = state_dset.known_iso()
    pop_dict = {}
    for col in state_iso.dataframe.columns:
        pop_dict[col] = population.loc[col,'POP']
        
    state_norm = state_iso.population_normalizer(pop_dict)

    # first idea: for every location give 0s to dates which have values less than average
    #             and 1s to dates greater than average
    zro = []
    for h in state_norm.dataframe.columns:
        arr = []
        locat = state_norm.dataframe.loc[:,h]
        start = locat.loc[locat>=locat.mean()].index[0]
        for g in locat.index:
            if g < start:
                arr.append(0)
            else:
                arr.append(1)
        zro.append(arr)

    binry = pd.DataFrame(zro).T
    binry.columns = state_norm.dataframe.columns
    binry.index = state_norm.dataframe.index

    g,n = links_initiators(binry,10000,10)

    g.to_csv("average_g.csv")
    n.to_csv("average_n.csv")
'''


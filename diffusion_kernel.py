import numpy as np
import math
from scipy.sparse.csgraph import laplacian
import pandas as pd
from matrix_operation import mat_opr
import matplotlib.pyplot as plt

state_dset = pd.read_csv('collected_data/state_dataset.csv', index_col = 0)
state_dset = mat_opr(state_dset)
s_adj = pd.read_csv("collected_data/state_adjacency.csv", index_col = 0)

state_adj = pd.DataFrame(columns = state_dset.dataframe.columns, index = state_dset.dataframe.columns)

for ind in s_adj.index:
    st = s_adj.loc[ind,'state']
    ad = s_adj.loc[ind, 'adj']
    dist = s_adj.loc[ind, 'distance']

    state_adj.loc[st,ad] = 1
state_adj = state_adj.replace(np.nan, 0)

L = np.array(laplacian(state_adj))
I = np.identity(len(L))

lambd = 1000
Kinv = I + lambd * L
K = np.linalg.inv(Kinv)

diff = pd.DataFrame(K, columns = state_adj.columns, index = state_adj.index)
diff.loc[:,'North Dakota'].plot()
plt.show()
'''
p = [   [-2, 1, 1, 0], 
        [1,-2,0,1], 
        [1,0,-2,1],
        [0,1,1,-2]  ]

L = -1 * np.array(p)

I = np.array([[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]])
'''

'''
lambdaa = 0.5

Kinv = I + lambdaa * L   
print(Kinv)
print()
K = np.linalg.inv(Kinv)
print(K)
'''

'''
w,v = np.linalg.eigh(L)

beta = 1
exper = []
for eig in w:
    exper.append(math.exp(eig * beta))

eig = np.diag(exper)
Kb = np.dot(v, np.dot(eig, np.transpose(v)))
print(Kb)
'''
'''
kb = np.zeros((4,4))
beta = 1
for i in range(len(s)):
    vi = vh[i]
    e = math.exp(beta * s[i])
    vit = np.transpose(vh)[i]

    adder = np.dot(vi * e, vit)
    kb += adder
    '''

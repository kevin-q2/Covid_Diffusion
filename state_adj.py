import os
import pandas as pd
from adj_list import *

def make_graph():
    if __name__ == '__main__':
        s_adj = pd.read_csv("collected_data/state_adjacency.csv", index_col = 0)
        state_case = pd.read_csv('collected_data/state_dataset.csv', index_col = 0)

    else:
        cwd = os.path.dirname(os.path.realpath(__file__))
        filer1 = os.path.join(cwd, 'collected_data/state_adjacency.csv')
        filer2 = os.path.join(cwd, 'collected_data/state_dataset.csv')
        s_adj = pd.read_csv(filer1, index_col = 0)
        state_case = pd.read_csv(filer2, index_col = 0)

    graph = Graph({}, [])
    for state in state_case.columns:
        s = s_adj.loc[s_adj.state == state]
        adjs = s.adj
        for a in adjs:
            dist = float(s.loc[s.adj == a].distance)
            graph.add_edge(state, a, w = dist)

        if len(adjs) == 0:
            graph.add_node(state)

    return graph
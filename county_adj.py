import os
import pandas as pd
from adj_list import *

# graph stuff: (adjacency list)
'''
class Node:
    def __init__(self,name, geo):
        self.name = name
        self.geo_id = geo
        self.surge_vals = {}
        self.neighbors = {}

    def add_neighbor(self, neighbor, weight):
        # where neighbor is a node
        self.neighbors[neighbor] = weight
        
    def get_adj(self):
        return self.neighbors.keys()

    
class Edge:
    def __init__(self, v1, v2, w):
        # First Node
        self.v1 = v1
        # Second Node
        self.v2 = v2
        # Weight
        self.w = w

class Graph:
    def __init__(self, vertices, edges):
        # vertices should be a dict of known vertices or an empty dict
        self.V = vertices
        # edges is just represented as a list of edges or initialized w/ empty list
        self.E = edges

    def add_node(self, node, geo):
        self.V[node] = Node(node, geo)

    def get_node(self, name):
        try:
            return self.V[name]
        except KeyError: 
            return None

    def get_edge(self, v1, v2):
        for e in self.E:
            if e.v1 == v1 and e.v2 == v2:
                return e
        

    def add_edge(self, v1, v2, w = 0, geo1 = None, geo2 = None):
        if v1 not in self.V.keys():
            self.add_node(v1, geo1)
        if v2 not in self.V.keys():
            self.add_node(v2, geo2)

        self.V[v1].add_neighbor(self.V[v2], w)
        self.E.append(Edge(v1,v2,w))
'''


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

abbrev = dict(map(reversed, us_state_abbrev.items()))


def read_county_adj(path, graph, county_dist, county_case):
    # takes the county adj text file and creates the adjacency list
    f = open(path)
    lines = f.readlines()

    # some words to remove ... makes it easier because Johns hopkins doesn't include these in their data
    reppers = [" County", " city", " Parish", " Municipality", " and", 
    " Borough", "Census Area"]

    curr = None
    #ab_frame = []

    for line in lines:
        mod = line.replace('"', '')
        mod = mod.replace('\n', '')
        mod = mod.split('\t')

        # a few cases to help me parse through the text file
        if mod[0] != '':

            try:
                # county lookup from county case data
                name1 = county_case.loc[int(mod[1]),:].index[0]
                graph.add_node(name1, geo_id = int(mod[1]))
                curr = (int(mod[1]), name1)
            except KeyError:
                curr = None

            
            #if int(mod[1]) != int(mod[3]):
            if curr is not None and int(mod[3]) != curr[0]:
                try:
                    #distance lookup from distance dataset
                    lookup = county_dist.loc[county_dist.county1 == int(mod[1])]
                    lookup = lookup.loc[lookup.county2 == int(mod[3])]

                    # for converting from original dataframe, only need this once
                    #ab_frame.append(lookup.values.tolist()[0])

                    w_i = float(lookup.mi_to_county)

                    name2 = county_case.loc[int(mod[3]),:].index[0]
                    graph.add_node(name2, int(mod[3]))
                    graph.add_edge(v1 = name1, v2 = name2, w = w_i) #, geo2 = int(mod[3]))

                except:
                    #print(name1, " and ", name2)
                    pass
                

        else:
            if curr is not None and int(mod[3]) != curr[0]:
                try:
                    name2 = county_case.loc[int(mod[3]),:].index[0]

                    lookup = county_dist.loc[county_dist.county1 == curr[0]]
                    lookup = lookup.loc[lookup.county2 == int(mod[3])]

                    # for converting from original dataframe, only need this once
                    #ab_frame.append(lookup.values.tolist()[0])

                    w_i = float(lookup.mi_to_county)
                    graph.add_node(name2, int(mod[3]))
                    graph.add_edge(v1 = curr[1], v2 = name2, w = w_i) #, geo2 = int(mod[3]))
                except:
                    #print(curr, " and ", name2)
                    pass

    #ab_frame = pd.DataFrame(ab_frame, columns = ['county1', 'mi_to_county', 'county2'])
    #print(ab_frame)
    #ab_frame.to_csv("collected_data/adj_county_distances.csv")



def make_graph(): 
    if __name__ == '__main__':
        graph = Graph({}, [])
        county_distance = pd.read_csv('collected_data/adj_county_distances.csv')
        county_case = pd.read_csv("collected_data/county_dataset.csv", index_col = [2,0,1])
        read_county_adj('collected_data/county_adjacency.txt', graph, county_distance, county_case)
        return graph

    else:
        # If this is being imported as a module in jupyter notebook from the analysis folder:
        cwd = os.path.dirname(os.path.realpath(__file__))
        filer1 = os.path.join(cwd, 'collected_data/county_adjacency.txt')
        filer2 = os.path.join(cwd, 'collected_data/adj_county_distances.csv')
        filer3 = os.path.join(cwd, 'collected_data/county_dataset.csv')
        graph = Graph({}, [])
        county_distance = pd.read_csv(filer2)
        county_case = pd.read_csv(filer3, index_col = [2,0,1])
        read_county_adj(filer1, graph, county_distance, county_case)
        return graph
    






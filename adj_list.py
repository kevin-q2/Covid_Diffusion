# structure for the adjacency list I will be using as a graph for states/counties/other regions

class Node:
    def __init__(self,name,geo_id = None):
        self.name = name
        self.geo_id = geo_id      # A geographical id number for the county (not used for states)
        self.val = 0            # given a value of 0 or 1 to describe its infected state 
        self.surge_vals = {}    # a dict of items that include each of the regions basis factors from NMF
        self.neighbors = {}     # dict of neighbors -- {name:distance}

    def add_neighbor(self, neighbor, weight):
        # add a neighbor to the neighbor dict
        self.neighbors[neighbor] = weight

    def find_neighbor(self, neighbor_name):
        # find a certain neighbor given name
        for n in self.neighbors:
            if n.name == neighbor_name:
                return n
        
        return None
    
class Edge:
    # a class for the edges (not totally sure if this is necessary but i did it anyways)
    def __init__(self, v1, v2, w):
        # First Node
        self.v1 = v1
        # Second Node
        self.v2 = v2
        # Weight
        self.w = w

class Graph:
    # The main graph object
    def __init__(self, vertices, edges):
        # vertices should be a dict of known vertices or an empty dict to initialize
        self.V = vertices
        # edges is a list of known edges or an empty list to initialize
        self.E = edges

    def add_node(self, node_name, geo_id = None):
        # add an initialized node to the list of vertices
        self.V[node_name] = Node(node_name, geo_id)

    def get_node(self, name):
        # find a node within the list of vertices
        try:
            return self.V[name]
        except KeyError: 
            return None

    def get_edge(self, v1, v2):
        # find an edge within the list of edges
        for e in self.E:
            if e.v1 == v1 and e.v2 == v2:
                return e
        

    def add_edge(self, v1, v2, w = 0):
        # add an edge between two given nodes with a given weight (or just 0 for unweighted)
        if v1 not in self.V.keys():
            self.add_node(v1)
        if v2 not in self.V.keys():
            self.add_node(v2)

        self.V[v1].add_neighbor(self.V[v2], w)
        self.E.append(Edge(v1,v2,w))

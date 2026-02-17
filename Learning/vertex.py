class Vertex:
    def __init__(self, value, adj : set):
        self.value = value
        self.adj = adj
        
    def add_adj(self, vtx : Vertex):
        self.adj.add(vtx)
    
    def get_degree(self):
        return len(self.adj)
    
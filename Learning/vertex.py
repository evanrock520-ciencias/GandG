class Vertex:
    def __init__(self, value, adj : set):
        self.value = value
        self.adj = adj
        
    def add_adj(self, vtx : Vertex):
        self.adj.add(vtx)
    
    def get_degree(self):
        return len(self.adj)
    
    def __eq__(self, other):
        return isinstance(other, Vertex) and self.value == other.value

    def __hash__(self):
        return hash(self.value)
    
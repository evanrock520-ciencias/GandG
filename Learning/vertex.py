import copy

class Vertex:
    def __init__(self, value, adj : set):
        self.value = value
        self.adj = adj
        
    def add_adj(self, vtx : Vertex):
        self.adj.add(vtx)
    
    def get_degree(self):
        return len(self.adj)
    
    def __repr__(self):
        return f"V{self.value}"
    
    def __eq__(self, other):
        return isinstance(other, Vertex) and self.value == other.value

    def __hash__(self):
        return hash(self.value)
    
    def __deepcopy__(self, memo):
        new_vtx = Vertex(copy.deepcopy(self.value, memo), set())
        memo[id(self)] = new_vtx
        new_vtx.adj = {memo.get(id(a), a) for a in self.adj}
        return new_vtx
        
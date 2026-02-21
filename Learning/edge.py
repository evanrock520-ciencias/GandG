from vertex import Vertex

class Edge:
    def __init__(self, vtx1 : Vertex, vtx2 : Vertex):
        self.vtx1 = vtx1
        self.vtx2 = vtx2
        
    def to_string(self) -> str:
        return f"({self.vtx1.value}, {self.vtx2.value})"
    
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        
        return (
            (self.vtx1 == other.vtx1 and self.vtx2 == other.vtx2) or
            (self.vtx1 == other.vtx2 and self.vtx2 == other.vtx1)
        )
    
    def __hash__(self):
        return hash(frozenset([self.vtx1, self.vtx2]))
        
    def __contains__(self, vtx):
        return vtx == self.vtx1 or vtx == self.vtx2
        
    def is_adj(self, edge : Edge):
        return (self.vtx1 in edge) != (self.vtx2 in edge)
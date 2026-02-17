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
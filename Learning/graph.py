from vertex import Vertex
from edge import Edge

class Graph:
    def __init__(self, vertices : list, edges : list):
        self.vertices = vertices
        self.edges = edges
        self.degrees = {vtx: vtx.get_degree() for vtx in vertices}
        
    def add_vertex(self, vtx : Vertex):
        self.vertices.append(vtx)
        self.degrees[vtx] = vtx.get_degree()
        
    def add_connection(self, idx1, idx2):
        edge = Edge(self.vertices[idx1], self.vertices[idx2])
        
        if edge in self.edges:
            print(f"The edge {edge.to_string()} is alredy in edges")
            return
        
        self.vertices[idx1].add_adj(self.vertices[idx2])
        self.vertices[idx2].add_adj(self.vertices[idx1])
        
        self.degrees[self.vertices[idx1]] += 1
        self.degrees[self.vertices[idx2]] += 1
        
        self.edges.append(edge)
        
    def get_order(self) -> int:
        return len(self.vertices)
    
    def get_size(self) -> int:
        return len(self.edges)
    
    def get_total_degree(self) -> int:
        total_degree = 0
        for vtx in self.vertices:
            total_degree += len(vtx.adj)
        
        return total_degree
    
    def get_gamma_degree(self) -> int:
        return min(self.degrees.values())
    
    def get_delta_degree(self) -> int:
        return max(self.degrees.values())
    
    def is_subgraph(self, graph : Graph):
        for vtx in graph.vertices:
            if vtx not in self.vertices:
                return False
            
        for edge in graph.edges:
            if edge not in self.edges:
                return False
            
        return True
    
    def delete_vertices(self, vertices_to_delete: list):
        for vtx in vertices_to_delete:
            if vtx in self.vertices:
                for neighbor in vtx.adj:
                    if neighbor in self.degrees:
                        self.degrees[neighbor] -= 1
                    if vtx in neighbor.adj:
                        neighbor.adj.remove(vtx)
                
                self.vertices.remove(vtx)
                self.degrees.pop(vtx, None)
                
                for edge in self.edges[:]:
                    if edge.vtx1 == vtx or edge.vtx2 == vtx:
                        print(f"Deleting {edge.to_string()}")
                        self.edges.remove(edge)
                        
    def count_same_degree(self, desired_degree: int) -> int:
        counter = 0
        for vtx_dg in self.degrees.values():
            if vtx_dg == desired_degree:
                counter += 1
        return counter
        
    def show_edges(self):
        for edge in self.edges:
            print(edge.to_string())
            
    def show_degrees(self):
        for vtx, dg in self.degrees.items():
            print((f"({vtx} : {dg})"))
            

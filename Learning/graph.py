import pandas as pd
import numpy as np

from vertex import Vertex
from edge import Edge

class Graph:
    def __init__(self, vertices : list, edges : list):
        self.vertices = vertices
        self.edges = edges
        self.degrees = {vtx: vtx.get_degree() for vtx in vertices}
        self.degree_sequence = None
        self.update_degree_sequence()
        
    def add_vertex(self, vtx : Vertex):
        self.vertices.append(vtx)
        self.degrees[vtx] = vtx.get_degree()
        self.update_degree_sequence()
        
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
        self.update_degree_sequence()
        
    def get_order(self) -> int:
        return len(self.vertices)
    
    def get_size(self) -> int:
        return len(self.edges)
    
    def get_total_degree(self) -> int:
        return sum(self.degree_sequence)
    
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
    
    def is_independent(self, vertices: list):
        for vtx in vertices:
            if vtx not in self.vertices:
                return False

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                v = vertices[i]
                u = vertices[j]

                if Edge(v, u) in self.edges:
                    return False

        return True
    
    def is_regular(self):
        if len(self.degree_sequence) == 0:
            return True
        
        fixed_value = self.degree_sequence[0]
        
        for value in self.degree_sequence:
            if value != fixed_value:
                return False
            
        return True
    
    def is_bipartite(self):
        color = {}
        for start in self.vertices:
            if start in color:
                continue
            
            color[start] = 0
            queue = [start]

            while queue:
                vtx = queue.pop(0)
                for utx in vtx.adj:
                    if utx not in color:
                        color[utx] = 1 - color[vtx]
                        queue.append(utx)
                    elif color[utx] == color[vtx]:
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
                        
        self.update_degree_sequence()
    
    def build_complement(self) -> Graph:
        cmp_vertices = self.vertices.copy()
        complement = Graph(cmp_vertices, [])
        for i, vtx in enumerate(cmp_vertices):
            for j, other in enumerate(cmp_vertices):
                if vtx == other:
                    continue
                
                cmp_edge = Edge(vtx, other)
                
                if cmp_edge not in self.edges:
                    complement.add_connection(i, j)
                    
        return complement
    
    def build_adj_matrix(self) -> pd.DataFrame:
        order = self.get_order()
        matrix = np.zeros((order, order))

        for i, vtx in enumerate(self.vertices):
            for j, utx in enumerate(self.vertices):
                if utx in vtx.adj:
                    matrix[i][j] = 1

        return pd.DataFrame(
            matrix,
            columns=[v.value for v in self.vertices],
            index=[v.value for v in self.vertices]
        )

    def update_degree_sequence(self):
        self.degree_sequence = sorted(self.degrees.values(), reverse=True)
                        
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
            
    def show_degree_sequence(self):
        sequence = '('
        for dg in self.degree_sequence:
            sequence += f'{dg},'
        sequence += ')'
        
        print(sequence)

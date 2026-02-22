import pandas as pd
import numpy as np
from itertools import combinations
import copy

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
        """
        Add a vertex to the graph.
        
        Args:
            vtx: The vertex we want to add to the graph.
            
        """
        self.vertices.append(vtx)
        self.degrees[vtx] = vtx.get_degree()
        self.update_degree_sequence()
        
    def add_connection(self, idx1, idx2):
        """
        Creates an edge with 2 vertex.
        
        Args:
                idx1: The index of the first vertex in self.vertices
                idx2: The index of the second vertex in self.vertices
            
        """
        edge = Edge(self.vertices[idx1], self.vertices[idx2])
        
        if edge in self.edges:
            # print(f"The edge {edge.to_string()} is alredy in edges")
            return
        
        self.vertices[idx1].add_adj(self.vertices[idx2])
        self.vertices[idx2].add_adj(self.vertices[idx1])
        
        self.degrees[self.vertices[idx1]] += 1
        self.degrees[self.vertices[idx2]] += 1
        
        self.edges.append(edge)
        self.update_degree_sequence()
        
    def get_order(self) -> int:
        """
        Returns:
            int: The order of the graph. How many vertex there are in the graph.
        """
        return len(self.vertices)
    
    def get_size(self) -> int:
        """
        Returns:
            int: The size of the graph. How many edges there are in the graph.
        """
        return len(self.edges)
    
    def get_total_degree(self) -> int:
        """
        Returns:
            int: The sum of the degree sequence.
        """
        return sum(self.degree_sequence)
    
    def get_gamma_degree(self) -> int:
        """
        Returns: 
            int: The min degree in the degree sequence of the graph.
        """
        return min(self.degree_sequence)
    
    def get_delta_degree(self) -> int:
        """
        Returns:
            int: The max degree in the degree sequence of the graph.
        """
        return max(self.degree_sequence)
    
    def is_subgraph(self, graph : Graph) -> bool:
        """
        Determines if the given graph is a subgraph of this graph.
        
        Args: 
            graph: The given graph.
        
        Returns:
            bool: If the given graph is a subgraph of this graph.
        """
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
    
    def bipartite_inducted_by(self, x_vertices: list, y_vertices: list) -> Graph:
        all_selected = set(x_vertices) | set(y_vertices)
        for vtx in all_selected:
            if vtx not in self.vertices:
                raise ValueError(f"The vertex {vtx.value} is not in the original graph.")

        set_x = set(x_vertices)
        set_y = set(y_vertices)

        vertices = x_vertices + y_vertices
        idxs = {vtx: i for i, vtx in enumerate(vertices)}
        bpt_graph = Graph(vertices, [])

        for vtx in vertices:
            u_idx = idxs[vtx]
            for adj in vtx.adj:
                if adj in all_selected:
                    if (vtx in set_x and adj in set_x) or (vtx in set_y and adj in set_y):
                        raise ValueError(f"This can't be bipartite: {vtx.value} and {adj.value} are in the same set.")
                    v_idx = idxs[adj]
                    bpt_graph.add_connection(u_idx, v_idx)

        return bpt_graph
    
    def is_walk(self, vertices : list) -> bool:
        for vtx in vertices:
            if vtx not in self.vertices:
                return False
            
        for idx in range(len(vertices) - 1):
            if vertices[idx + 1] not in vertices[idx].adj:
                return False
            
        return True
    
    def is_cycle(self, vertices: list):
        if not self.is_walk(vertices):
            return False
        
        if vertices[0] != vertices[-1]:
            return False

        return len(vertices[:-1]) == len(set(vertices[:-1]))
    
    def is_trail(self, vertices: list) -> bool:
        if not self.is_walk(vertices):
            return False
        edges = [Edge(vertices[idx], vertices[idx + 1]) for idx in range(len(vertices) - 1)]
        return len(edges) == len(set(edges))

    def is_path(self, vertices: list) -> bool:
        if not self.is_walk(vertices):
            return False
        return len(vertices) == len(set(vertices)) 
    
    def delete_vertices(self, vertices_to_delete: list):
        """
        Removes the specified vertices from the graph, along with their 
        incident edges.

        Args:
            vertices_to_delete: A list of vertices to be removed from this graph.
        """
        for vtx in vertices_to_delete:
            if vtx in self.vertices:
                neighbors = list(vtx.adj)
                for neighbor in neighbors:
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
        cmp_vertices = copy.deepcopy(self.vertices)
        complement = Graph(cmp_vertices, [])
        for i, vtx in enumerate(cmp_vertices):
            for j, other in enumerate(cmp_vertices):
                if vtx == other:
                    continue
                
                cmp_edge = Edge(vtx, other)
                
                if cmp_edge not in self.edges:
                    complement.add_connection(i, j)
                    
        return complement
    
    def inducted_by(self, vertices: list) -> Graph:
        for vtx in vertices:
            if vtx not in self.vertices:
                raise ValueError()

        edges = set()
        for vtx in vertices:
            for adj in vtx.adj:
                if adj in vertices:
                    edges.add(Edge(vtx, adj))

        copied = {vtx: copy.deepcopy(vtx) for vtx in vertices}
        for original, copy_vtx in copied.items():
            copy_vtx.adj = {copied[a] for a in original.adj if a in copied}

        new_vertices = list(copied.values())
        new_edges = [Edge(copied[edg.vtx1], copied[edg.vtx2]) for edg in edges]

        return Graph(new_vertices, new_edges)
        
    def lines_graph(self) -> Graph:
        line_vertices = [Vertex(edge, set()) for edge in self.edges]
        graph = Graph(line_vertices, [])
        
        degree = len(line_vertices)
        
        for idx in range(degree):
            for jdx in range(idx + 1, degree):
                vtx_i = line_vertices[idx]
                vtx_j = line_vertices[jdx]
                
                if vtx_i.value.is_adj(vtx_j.value):
                    graph.add_connection(idx, jdx)
                    
        return graph

        
    def build_adj_matrix(self) -> pd.DataFrame:
        order = self.get_order()
        matrix = np.zeros((order, order))

        index = {vtx: idx for idx, vtx in enumerate(self.vertices)}

        for vtx in self.vertices:
            idx = index[vtx]
            for utx in vtx.adj:
                jdx = index[utx]
                matrix[idx][jdx] = 1

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

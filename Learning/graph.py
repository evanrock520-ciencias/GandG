import pandas as pd
import numpy as np
from itertools import combinations
import copy

from vertex import Vertex
from edge import Edge

class Graph:
    def __init__(self, vertices : list, edges : list):
        self._vertices = vertices
        self._edges = edges
        
        for edge in self._edges:
            if edge.vtx1 not in self._vertices or edge.vtx2 not in self._vertices:
                raise ValueError("There are edges with invalid vertices.")
    
        self._degrees = {vtx: vtx.get_degree() for vtx in vertices}
        self._degree_sequence = None
        self._update_degree_sequence()
        
    @property
    def vertices(self):
        """The vertices of the graph"""
        return list(self._vertices)
    
    @property
    def edges(self):
        """The edges of the graph"""
        return list(self._edges)
    
    @property
    def degrees(self):
        """The degrees of every vertex"""
        return dict(self._degrees)
    
    @property
    def degree_sequence(self):
        """The degrees sequence of the graph"""
        self._update_degree_sequence()
        return list(self._degree_sequence)
    
    @property
    def order(self):
        """The order of the graph. How many vertex there are in the graph."""
        return len(self._vertices)
    
    @property
    def size(self):
        """The size of the graph. How many edges there are in the graph."""
        return len(self._edges)
    
    def _update_degree_sequence(self):
        """Updates the degree sequence with the degrees dict values."""
        self._degree_sequence = sorted(self._degrees.values(), reverse=True)
        
    def _remove_edge(self, edge : Edge):
        """
        Removes an edge from the graph.
        Args:
            edge: The edge we want to remove from the graph.
        """
        self._degrees[edge.vtx1] -= 1
        self._degrees[edge.vtx2] -= 1
                
        edge.vtx1.adj.remove(edge.vtx2)
        edge.vtx2.adj.remove(edge.vtx1)
                
        self._edges.remove(edge)
        self._update_degree_sequence()
        
    def add_vertex(self, vtx : Vertex):
        """
        Add a vertex to the graph.
        Args:
            vtx: The vertex we want to add to the graph.
            
        """
        vtx.adj.clear()
        self._vertices.append(vtx)
        self._degrees[vtx] = vtx.get_degree()
        self._update_degree_sequence()
        
    def add_connection(self, a, b):
        """
        Creates an edge with 2 vertex.
        Args:
            a: Could be a index to find a vertex in the graph vertices or a vertex itself.
            b: Could be a index to find a vertex in the graph vertices or a vertex itself.
        """
        vtx = self._vertices[a] if isinstance(a, int) else a
        utx = self._vertices[b] if isinstance(b, int) else b

        edge = Edge(vtx, utx)
            
        if edge in self._edges:
            return
        
        vtx.add_adj(utx)
        utx.add_adj(vtx)
        
        self._degrees[vtx] += 1
        self._degrees[utx] += 1
        
        self._edges.append(edge)
        self._update_degree_sequence()
    
    def get_total_degree(self) -> int:
        """
        Returns:
            int: The sum of the degree sequence.
        """
        return sum(self._degree_sequence)
    
    def get_gamma_degree(self) -> int:
        """
        Returns: 
            int: The min degree in the degree sequence of the graph.
        """
        return min(self._degree_sequence)
    
    def get_delta_degree(self) -> int:
        """
        Returns:
            int: The max degree in the degree sequence of the graph.
        """
        return max(self._degree_sequence)
    
    def is_subgraph(self, graph : Graph) -> bool:
        """
        Determines if the given graph is a subgraph of this graph.
        Args: 
            graph: The given graph.
        Returns:
            bool: If the given graph is a subgraph of this graph.
        """
        for vtx in graph._vertices:
            if vtx not in self._vertices:
                return False
            
        for edge in graph._edges:
            if edge not in self._edges:
                return False
            
        return True
    
    def is_independent(self, vertices: list):
        for vtx in vertices:
            if vtx not in self._vertices:
                return False

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                v = vertices[i]
                u = vertices[j]

                if Edge(v, u) in self._edges:
                    return False

        return True
    
    def is_regular(self):
        """
        Determines if all the vertices have the same degree.
        """
        if len(self._degree_sequence) == 0:
            return True
        
        fixed_value = self._degree_sequence[0]
        
        for value in self._degree_sequence:
            if value != fixed_value:
                return False
            
        return True
    
    def is_complete(self):
        """Determines if the graph has all the posible combinations of edges."""
        order = self.order
        return self.size == (order * (order - 1)) // 2
    
    def is_bipartite(self):
        """Determines if the graph could be bipartite."""
        return self.get_bipartition() is not None
    
    def get_bipartition(self) -> tuple:
        color = {}
        for start in self._vertices:
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
                        return None
        
        X = [vtx for vtx in self._vertices if color[vtx] == 0]
        Y = [vtx for vtx in self._vertices if color[vtx] == 1]
        return (X, Y)
    
    def is_bipartite_complete(self) -> bool:
        if not self.is_bipartite():
            return False
        
        X, Y = self.get_bipartition()
        for vtx in X:
            for utx in Y:
                if not Edge(vtx, utx) in self._edges:
                    return False
                
        return True
    
    def bipartite_inducted_by(self, x_vertices: list, y_vertices: list) -> Graph:
        all_selected = set(x_vertices) | set(y_vertices)
        for vtx in all_selected:
            if vtx not in self._vertices:
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
            if vtx not in self._vertices:
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
    
    def is_connected(self):
        if len(self._vertices) == 0:
            return True
        
        visited = set()
        queue = [self._vertices[0]]
        visited.add(self._vertices[0])
        
        while queue:
            vtx = queue.pop(0)
            for adj in vtx.adj:
                if adj not in visited:
                    visited.add(adj)
                    queue.append(adj)
                    
        return len(visited) == len(self._vertices)
    
    def is_unconnected(self):
        return not self.is_connected()
    
    def delete_vertices(self, vertices_to_delete: list):
        """
        Removes the specified vertices from the graph, along with their 
        incident edges.

        Args:
            vertices_to_delete: A list of vertices to be removed from this graph.
        """
        for vtx in vertices_to_delete:
            if vtx in self._vertices:
                neighbors = list(vtx.adj)
                for neighbor in neighbors:
                    if neighbor in self._degrees:
                        self._degrees[neighbor] -= 1
                    if vtx in neighbor.adj:
                        neighbor.adj.remove(vtx)
                
                self._vertices.remove(vtx)
                self._degrees.pop(vtx, None)
                
                for edge in self._edges[:]:
                    if vtx in edge:
                        self._edges.remove(edge)
                        
        self._update_degree_sequence()
        
    def delete_edges(self, edges_to_delete : list):
        for edge in edges_to_delete:
            if edge in self._edges:
                self._remove_edge(edge)
            
    def build_complement(self) -> Graph:
        cmp_vertices = copy.deepcopy(self._vertices)
        complement = Graph(cmp_vertices, [])
        for i, vtx in enumerate(cmp_vertices):
            for j, other in enumerate(cmp_vertices):
                if vtx == other:
                    continue
                
                cmp_edge = Edge(vtx, other)
                
                if cmp_edge not in self._edges:
                    complement.add_connection(i, j)
                    
        return complement
    
    def inducted_by(self, vertices: list) -> Graph:
        for vtx in vertices:
            if vtx not in self._vertices:
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
        line_vertices = [Vertex(edge, set()) for edge in self._edges]
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
        order = self.order
        matrix = np.zeros((order, order), dtype=int)

        index = {vtx: idx for idx, vtx in enumerate(self._vertices)}

        for vtx in self._vertices:
            idx = index[vtx]
            for utx in vtx.adj:
                jdx = index[utx]
                matrix[idx][jdx] = 1

        return pd.DataFrame(
            matrix,
            columns=[vtx.value for vtx in self._vertices],
            index=[vtx.value for vtx in self._vertices]
        )        
        
    def build_inc_matrix(self) -> pd.DataFrame:
        order = self.order
        size = self.size

        matrix = np.zeros((order, size), dtype=int)

        vtx_index = {vtx: i for i, vtx in enumerate(self._vertices)}

        for eidx, edge in enumerate(self._edges):
            for vtx in edge:  
                vidx = vtx_index[vtx]
                matrix[vidx, eidx] = 1

        return pd.DataFrame(
            matrix,
            columns=[str(edge) for edge in self._edges],
            index=[vtx.value for vtx in self._vertices]
        )
        
    def distance_between(self, vtx: Vertex, utx: Vertex) -> int:
        if vtx == utx:
            return 0
        
        visited = {vtx}
        queue = [(vtx, 0)]
        
        while queue:
            wtx, distance = queue.pop(0)
            for adj in wtx.adj:
                if adj == utx:
                    return distance + 1
                
                if adj not in visited:
                    visited.add(adj)
                    queue.append((adj, distance + 1))
        
        return float('inf')
    
    def diameter(self) -> int:
        if not self.is_connected():
            return float('inf')
        return max(self.eccentricity(vtx) for vtx in self._vertices)
    
    def radius(self) -> int:
        if not self.is_connected():
            return float('inf')
        return min(self.eccentricity(vtx) for vtx in self._vertices)
    
    def eccentricity(self, vtx: Vertex) -> int:
        max_dist = 0
        for utx in self._vertices:
            max_dist = max(max_dist, self.distance_between(vtx, utx))
        return max_dist
                        
    def count_same_degree(self, desired_degree: int) -> int:
        counter = 0
        for vtx_dg in self._degrees.values():
            if vtx_dg == desired_degree:
                counter += 1
        return counter
    
    def is_central(self, vtx: Vertex) -> bool:
        return self.eccentricity(vtx) == self.radius()
    
    def is_peripheral(self, vtx: Vertex):
        return self.eccentricity(vtx) == self.diameter()
    
    def center(self) -> list:
        return [vtx for vtx in self._vertices if self.is_central(vtx)]

    def periphery(self) -> list:
        return [vtx for vtx in self._vertices if self.is_peripheral(vtx)]
    
    def get_vertex(self, value):
        return next((vtx for vtx in self._vertices if vtx.value == value), None)
    
    def is_cutoff_point(self, vtx: Vertex) -> bool:
        if not self.is_connected():
            return False
        cutted_graph = self.clone()
        cutted_graph.delete_vertices([cutted_graph._vertices[self._vertices.index(vtx)]])
        return not cutted_graph.is_connected()
    
    def cut_vertices(self) -> list:
        return [vtx for vtx in self._vertices if self.is_cutoff_point(vtx)]

    def is_bridge(self, edge: Edge) -> bool:
        if not self.is_connected():
            return False
        cutted_graph = self.clone()
        idx = self._edges.index(edge)
        cutted_graph.delete_edges([cutted_graph._edges[idx]])
        return not cutted_graph.is_connected()
        
    def clone(self) -> Graph:
        return copy.deepcopy(self)
        
    def show_edges(self):
        for edge in self._edges:
            print(edge.to_string())
            
    def show_degrees(self):
        for vtx, dg in self._degrees.items():
            print((f"({vtx} : {dg})"))
            
    def show_degree_sequence(self):
        sequence = '('
        for dg in self._degree_sequence:
            sequence += f'{dg},'
        sequence += ')'
        
        print(sequence)

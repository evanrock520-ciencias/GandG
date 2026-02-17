from graph import Graph
from graph import Vertex
import math

def main(vertices : list):
    graph = Graph(vertices, [])
    
    for idx1, vtx in enumerate(vertices):
        for idx2, other in enumerate(vertices):
            if vtx == other:
                continue
            
            if vtx.value % 2 == 0:
                if is_prime(other.value) and vtx.value < other.value:
                    graph.add_connection(idx1, idx2)
                    
    graph.show_edges()
    print(f"The size of the graph is {graph.get_size()}")
    print(f"The min degree of the graph {graph.get_gamma_degree()}")
    print(f"The max degree of the graph {graph.get_delta_degree()}")
    print(f"There are {graph.count_same_degree(4)} vertex with the same degree")

            
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    max_divisor = int(math.sqrt(n))
    for i in range(3, max_divisor + 1, 2):
        if n % i == 0:
            return False
            
    return True

def create_vertices(end : int) -> list:
    vertices = []
    for i in range(1, end):
        vertices.append(Vertex(i, set()))
        
    return vertices

vtxs = create_vertices(40)

main(vtxs)
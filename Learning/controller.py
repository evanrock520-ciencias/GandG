from graph import Graph, Vertex, Edge

class Controller:
    @staticmethod
    def is_graphicable(degrees: tuple) -> bool:
        if sum(degrees) % 2 != 0:
            return False
            
        odds = [dg for dg in degrees if dg % 2 != 0] 
        if len(odds) % 2 != 0:
            return False
            
        return Controller.havel_hakimi(list(degrees))
        
    @staticmethod
    def havel_hakimi(degrees: list) -> bool:
        degrees = sorted([dg for dg in degrees if dg > 0], reverse=True)
            
        if not degrees:
            return True
            
        higher = degrees.pop(0)
        if higher > len(degrees):
            return False
            
        for idx in range(higher):
            degrees[idx] -= 1               
            if degrees[idx] < 0:
                return False
                
        return Controller.havel_hakimi(degrees)
    
    @staticmethod
    def graph_from_ds(degrees: tuple) -> Graph:
        if not Controller.is_graphicable(degrees):
            raise ValueError("The degrees sequence is not graphicable")
        
        seq = [{"id": idx, "degree": dg} for idx, dg in enumerate(degrees) if dg > 0]
        
        vertices = [Vertex(idx, set()) for idx in range(len(degrees))]
        graph = Graph(vertices, [])

        while seq:
            seq.sort(key=lambda x: x['degree'], reverse=True)
            
            current = seq.pop(0)
            dg = current['degree']
            
            for idx in range(dg):
                target = seq[idx]
                target['degree'] -= 1
                
                graph.add_connection(current['id'], target['id'])
                
            seq = [item for item in seq if item['degree'] > 0]

        return graph
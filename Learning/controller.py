class Controler:
    def is_graphicable(self, degrees: tuple) -> bool:
        if sum(degrees) % 2 != 0:
            return False
        
        odds = [dg for dg in degrees if dg % 2 != 0] 
        if len(odds) % 2 != 0:
            return False
        
        return self.havel_hakimi(list(degrees))
    
    def havel_hakimi(self, degrees: list) -> bool:
        degrees = sorted([dg for dg in degrees if dg > 0], reverse=True)
        
        if not degrees:
            return True
        
        higher = degrees.pop(0)
        if higher > len(degrees):
            return False
        
        for i in range(higher):
            degrees[i] -= 1
            if degrees[i] < 0:
                return False
            
        return self.havel_hakimi(degrees)
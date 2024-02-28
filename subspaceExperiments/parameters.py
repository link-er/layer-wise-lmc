import numpy

class Parameters:

    def __init__(self):

        raise  NotImplementedError

    def set(self):

        raise NotImplementedError

    def get(self):
         
        raise NotImplementedError
    
    def add(self, other):
         
        raise NotImplementedError
    
    def scalarMultiply(self, scalar):
         
        raise NotImplementedError
    
    def distance(self, other):
         
        raise NotImplementedError

    def getCopy(self):
         
        raise NotImplementedError
    
    def toVector(self) -> numpy.array:
         
        raise NotImplementedError
    
    def fromVector(self, v : numpy.array):
         
        raise NotImplementedError
    

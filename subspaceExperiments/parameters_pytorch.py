from parameters import Parameters
import numpy as np
from _collections import OrderedDict

class PyTorchNNParameters(Parameters):

    def __init__(self, stateDict : dict):
        self._state = stateDict
        self._shapes = OrderedDict()
        for k in self._state:
            arr = self._state[k]
            self._shapes[k] = arr.shape

    def set(self, stateDict: dict):
        if not isinstance(stateDict, dict):
            raise ValueError("Weights for PyTorchNNParameters should be given as python dictionary. Instead, the type given is " + str(type(stateDict)))
            
        self._state = stateDict
        self._shapes = OrderedDict()
        for k in self._state:
            self._shapes[k] = self._state[k].shape

        # to use it inline
        return self

    def get(self) -> dict:
        return self._state
    
    def add(self, other):
        if not isinstance(other, PyTorchNNParameters):
            error_text = "The argument other is not of type" + str(PyTorchNNParameters) + "it is of type " + str(type(other))
            self.error(error_text)
            raise ValueError(error_text)

        otherW = other.get()
        if set(self._state.keys()) != set(otherW.keys()):
            raise ValueError("Error in addition: state dictionary have different keys. This: "+str(set(self._state.keys()))+", other: "+str(set(otherW.keys()))+".")
        
        for k,v in otherW.items():
            self._state[k] = np.add(self._state[k], v)
    
    def scalarMultiply(self, scalar: float):
        if not isinstance(scalar, float):
            raise ValueError("Scalar should be float but is " + str(type(scalar)) + ".")
        
        for k in self._state:
            if isinstance(self._state[k], np.int64):
                self._state[k] *= int(scalar)
            else:
                self._state[k] = np.multiply(self._state[k], scalar, out=self._state[k], casting="unsafe")

    def addNormalNoise(self, loc, scale):
        for k in self._state:
            if isinstance(self._state[k], np.int64):
                self._state[k] += int(np.random.normal(loc=loc, scale=scale))
            else:
                self._state[k] += np.random.normal(loc=loc, scale=scale, size=self._state[k].shape)

    
    def distance(self, other) -> float:
        if not isinstance(other, PyTorchNNParameters):
            error_text = "The argument other is not of type" + str(PyTorchNNParameters) + "it is of type " + str(type(other))
            self.error(error_text)
            raise ValueError(error_text)

        otherW = other.get()
        if set(self._state.keys()) != set(otherW.keys()):
            raise ValueError("Error in addition: state dictionary have different keys. This: "+str(set(self._state.keys()))+", other: "+str(set(otherW.keys()))+".")
        
        w1 = self.flatten()
        w2 = other.flatten() #instead of otherW, because otherW is of type np.array instead of paramaters
        dist = np.linalg.norm(w1-w2)
        
        return dist
    
    def flatten(self) -> np.ndarray:
        flatParams = []
        for k in self._state:
            flatParams += np.ravel(self._state[k]).tolist()
        return np.asarray(flatParams)
    
    def getCopy(self):
        newState = OrderedDict()

        for k in self._state:
            newState[k] = self._state[k].copy()
        newParams = PyTorchNNParameters(newState)
        return newParams

    def toVector(self)->np.array:
        return self.flatten()

    def fromVector(self, v:np.array):
        currPos = 0
        newState = OrderedDict()
        for k in self._shapes: #shapes contains the shapes of all weight matrices in the model and all the additional parameters, e.g., batch norm
            s = self._shapes[k]
            n = np.prod(s) #the number of elements n in the curent weight matrix
            arr = v[currPos:currPos+n].reshape(s)
            newState[k] = arr.copy()
            currPos += n
        self.set(newState)

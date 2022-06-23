import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.shape = np.shape(input_tensor)
        return np.reshape(input_tensor, (self.shape[0], -1))

    def backward(self,error_tensor):
        return np.reshape(error_tensor, self.shape)

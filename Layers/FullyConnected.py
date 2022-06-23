import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.gradient_weights = None

        self._weights = np.random.rand(input_size + 1, output_size)
        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property         
    def weights(self):        
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((weights, bias))

    def forward(self,input_tensor):
        self.input_tensor = np.insert(input_tensor, input_tensor.shape[1], np.ones(input_tensor.shape[0]), axis = 1)
        return np.dot(self.input_tensor, self.weights)

    def backward(self,error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return np.delete(np.dot(error_tensor, self.weights.T), (-1), axis = 1)

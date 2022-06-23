from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        input_tensor_exp = np.exp(input_tensor - np.max(input_tensor))
        self.prediction_tensor = input_tensor_exp / input_tensor_exp.sum(axis=1, keepdims=True)
        return self.prediction_tensor

    def backward(self, error_tensor):
        sum = np.sum(error_tensor * self.prediction_tensor, axis=1, keepdims=True)
        return self.prediction_tensor * (error_tensor - sum)

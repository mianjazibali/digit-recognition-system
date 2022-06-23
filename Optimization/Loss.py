import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.prediction_tensor = input_tensor
        loss_tensor = -np.log(input_tensor + np.finfo(float).eps)
        return np.sum(loss_tensor[label_tensor == 1])

    def backward(self, label_tensor):
        return -(label_tensor / self.prediction_tensor)

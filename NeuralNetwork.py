from copy import deepcopy

class NeuralNetwork:
    def __init__(self, optimizer, weight_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        return self.loss_layer.forward(self.test(self.input_tensor), self.label_tensor)

    def backward(self):
        output = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            output = layer.backward(output)

        return output

    def append_layer(self, layer):
        if hasattr(layer, 'initialize'):
            layer.initialize(self.weight_initializer, self.bias_initializer)

        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)

        return self.layers.append(layer)

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor

    def train(self, iterations) -> None:
        for i in range(0, iterations):
            self.loss.append(self.forward())
            self.backward()

        return self.loss

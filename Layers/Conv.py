import math
import numpy as np
import scipy.signal
import copy

from .Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()

        self.trainable = True
        self.weights = np.random.rand(num_kernels, *convolution_shape)

        self.bias = np.random.random(num_kernels)
        self.num_kernels = num_kernels
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape

        self._optimizer = None
        self._gradient_weights = 0
        self._gradient_bias = 0

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(
            self.weights.shape,
            np.prod(self.convolution_shape),
            self.num_kernels * np.prod(self.convolution_shape[1:])
        )

        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = np.shape(input_tensor)[0]

        output_tensor = []
        for b in range(batch_size):
            _output_tensor = []
            for k in range(self.num_kernels):
                output = scipy.signal.correlate(input_tensor[b], self.weights[k], 'same')
                output = output[math.floor(output.shape[0] / 2)] + self.bias[k]

                if (len(self.stride_shape) == 1):
                    _output_tensor.append(output[::self.stride_shape[0]])
                elif (len(self.stride_shape) == 2):
                    _output_tensor.append(output[::self.stride_shape[0], ::self.stride_shape[1]])

            output_tensor.append(_output_tensor)

        return np.array(output_tensor)

    def backward(self,error_tensor):
        batch_size = np.shape(error_tensor)[0]
        num_kernels = self.convolution_shape[0]
        num_channels = self.num_kernels

        weights = np.swapaxes(self.weights, 1, 0)
        weights = np.flip(weights, 1)
        output_tensor =[]
        update_error_batch = np.zeros((batch_size, num_channels, *self.input_tensor.shape[2:]))
        for b in range(batch_size):
            if (len(self.stride_shape) == 1):
                update_error_batch[:, :, ::self.stride_shape[0]] = error_tensor[b]
            else:
                update_error_batch[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b]

            output_tensor_conv = []
            for n_k in range(num_kernels):
                output = scipy.signal.convolve(update_error_batch[b], weights[n_k], 'same')
                output = output[output.shape[0] // 2]
                output_tensor_conv.append(output)

            output_tensor.append(output_tensor_conv)

        self._gradient_weights = self.update_gradient_weights( error_tensor)
        if self.optimizer:
            self.weights = copy.deepcopy(self.optimizer).calculate_update(self.weights, self._gradient_weights)
            self.bias = copy.deepcopy(self.optimizer).calculate_update(self.bias, self._gradient_bias)

        return np.array(output_tensor)

    def update_gradient_weights(self,error_tensor):
        batch_size = np.shape(error_tensor)[0]
        temp_weights = 0
        num_channels = self.num_kernels
        update_error_batch = np.zeros((batch_size, num_channels, *self.input_tensor.shape[2:]))
        for b in range(batch_size):
            if (len(self.stride_shape) == 1):
                update_error_batch[:, :, ::self.stride_shape[0]] = error_tensor[b]
                self._gradient_bias = np.sum(error_tensor, axis=(0, 2))
                pading_X = np.pad(
                    self.input_tensor[b],
                    ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2)),
                    'constant',
                    constant_values = 0
                )
            else:
                update_error_batch[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b]
                self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
                pading_X = np.pad(
                    self.input_tensor[b],
                    (
                        (0, 0),
                        (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2),
                        (self.convolution_shape[2] // 2, (self.convolution_shape[2] - 1) // 2)
                    ),
                    'constant',
                    constant_values = 0
                )
            all_gradient_kernels = []
            for c in range(num_channels):
                each_gradient_kernel = []
                for channel_X in range(self.input_tensor.shape[1]):
                    gradient_weight = scipy.signal.correlate(pading_X[channel_X], update_error_batch[b][c], 'valid')
                    each_gradient_kernel.append(gradient_weight)

                all_gradient_kernels.append(each_gradient_kernel)

            all_gradient_kernels = np.array(all_gradient_kernels)
            temp_weights += all_gradient_kernels

        return temp_weights

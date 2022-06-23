import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
  
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channel_num, height_conv, width_conv = np.shape(input_tensor)

        pooling_layer_height = int(((height_conv - self.pooling_shape[0]) / self.stride_shape[0]) + 1)
        pooling_layer_width = int(((width_conv - self.pooling_shape[1]) / self.stride_shape[1]) + 1)

        max_values_list = []
        self.index_list_max_values = []
        for b in range(batch_size):
            for c in range(channel_num):
                for h in range(pooling_layer_height):
                    for w in range(pooling_layer_width):
                        pooling = input_tensor[
                            b,
                            c,
                            h * self.stride_shape[0] : h * self.stride_shape[0] + self.pooling_shape[0],
                            w * self.stride_shape[1] : w * self.stride_shape[1] + self.pooling_shape[1]
                        ]
                        max_values_list.append(np.max(pooling))                        
                        each_maxvalue_index = np.unravel_index(pooling.argmax(), pooling.shape)
                        self.index_list_max_values.append([
                            b,
                            c,
                            h * self.stride_shape[0] + each_maxvalue_index[0], 
                            w * self.stride_shape[1] + each_maxvalue_index[1]
                        ])

        return np.reshape(max_values_list, (batch_size, channel_num, pooling_layer_height, pooling_layer_width))

    def backward(self, error_tensor):      
        batch_size, channel_num, height_poolinglayer, width_poolinglayer = np.shape(error_tensor)
        conv = np.zeros(self.input_tensor.shape)

        i = 0
        for b in range(batch_size):
            for c in range(channel_num):
                for h in range(height_poolinglayer):
                    for w in range(width_poolinglayer):                          
                        index = self.index_list_max_values[i]                       
                        conv[tuple(index)] += error_tensor[b, c, h, w]                          
                        i += 1   
 
        return conv    

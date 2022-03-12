from src import Neuron
import numpy as np
from typing import *


class MaxPoolingLayer:
    def __init__(self, kernel_size: int, input_channels: int,
                 input_dimensions: Sequence[int]):
        """
        Initializes a max pooling layer.
        :param input_channels: Number of input channels (input width
        :param input_dimensions: Dimensions of input (height, width)
        """
        self.name = "MaxPoolingLayer"
        self.kernel_size = kernel_size
        self.input_dimensions = input_dimensions  # Dimensions of input (height, width)
        self.input_channels = input_channels  # Number of input channels
        self.output_channels = input_channels  # Number of output channels
        self.num_kernels = input_channels  # Number of output channels
        self.output_size = (input_dimensions[0] - kernel_size + 1,
                            input_dimensions[0] - kernel_size + 1)
        self.selected_vals = []

    def calculate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the layer.
        :param inputs: Inputs to the layer
        :return: Output of the layer
        """
        # TODO: keep the position of the max value for use in the unpooling step
        if self.input_channels == 1:  # Shape is (kernel_size x kernel_size)
            inputs = inputs.copy()[np.newaxis, ...]  # Add channel dimension
        outputs = []
        self.selected_vals = []
        for channel_ind in range(self.input_channels):
            kernel_output = []
            for kernel_x in range(self.output_size[0]):
                kernel_x_output = []
                for kernel_y in range(self.output_size[1]):
                    # Shape is (num_channels, kernel_size x kernel_size)
                    inputs_to_neuron = inputs[channel_ind,
                                       kernel_x:kernel_x + self.kernel_size,
                                       kernel_y:kernel_y + self.kernel_size] \
                        .reshape((self.input_channels, -1))
                    max_value = inputs_to_neuron.max()
                    max_index = inputs_to_neuron.argmax(axis=0)
                    max_index[0] = max_index[0]+kernel_x
                    max_index[1] = max_index[1] + kernel_y
                    max_index = np.append(channel_ind,max_index)
                    # Calculate the max value of the patch
                    self.selected_vals.append(max_index)
                    kernel_x_output.append(max_value)
                kernel_output.append(kernel_x_output)
            outputs.append(kernel_output)
        return np.array(outputs)

    #@staticmethod
    def calculate_wdeltas(self, wdeltas_next: List) -> List:
        """
        Calculates the weight deltas of the layer.
        :param wdeltas_next: Weight deltas of the next layer
        :return: Weight deltas of the layer
        """
        wdeltas = np.zeros((self.input_channels, self.input_dimensions[0], self.input_dimensions[1]))
        count =0
        wdeltas_next_flattened = np.array(wdeltas_next).flatten()
        for indexs in self.selected_vals:
            wdeltas[indexs[0],indexs[1],indexs[2]] = wdeltas_next_flattened[count]+wdeltas[indexs[0],indexs[1],indexs[2]]
            count = count+1

        return wdeltas

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
        self.output_size = (input_dimensions[0] - kernel_size + 1,
                            input_dimensions[0] - kernel_size + 1)

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
                    # Calculate the max value of the patch
                    max_value = inputs_to_neuron.max()
                    kernel_x_output.append(max_value)
                kernel_output.append(kernel_x_output)
            outputs.append(kernel_output)
        return np.array(outputs)

    @staticmethod
    def calculate_wdeltas(wdeltas_next: List) -> List:
        """
        Calculates the weight deltas of the layer.
        :param wdeltas_next: Weight deltas of the next layer
        :return: Weight deltas of the layer
        """
        return wdeltas_next

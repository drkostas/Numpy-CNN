from src import Neuron
import numpy as np
from typing import *


class FlattenLayer:
    def __init__(self, num_inputs: int):
        """
        Initializes a flatten layer.
        :param num_inputs: Number of inputs to each neuron
        """
        self.name = "FlattenLayer"
        self.num_inputs = num_inputs
        self.neurons_per_layer = num_inputs
        self.output_channels = 1
        self.input_shape = None

    def calculate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the layer.
        :param inputs: Inputs to the layer
        :return: Output of the layer
        """
        # TODO: keep info about the original shape of the input for the backpropagation
        self.input_shape = inputs.shape
        inputs_flatten = np.array(inputs).flatten()
        return inputs_flatten

    def calculate_wdeltas(self, wdeltas_next: np.ndarray) -> np.ndarray:
        """
        Calculates the weight deltas of the layer.
        :param wdeltas_next: Weight deltas of the next layer
        :return: Weight deltas of the layer
        """
        return wdeltas_next.reshape(self.input_shape)

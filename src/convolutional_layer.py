from src import Neuron
import numpy as np
from typing import *


class ConvolutionalLayer:
    def __init__(self, num_kernels: int, kernel_size: int, activation: str, input_dimensions: int, lr: float,
                 weights: np.ndarray):
        """
        Initializes a fully connected layer.
        :param neurons_per_layer: Number of neurons in the layer
        :param activation: Activation function
        :param num_inputs: Number of inputs to each neuron
        :param lr: Learning rate
        :param weights: Weights of the layer
        """
        self.num_kernels = num_kernels
        self.kernel_size = self.kernel_size
        self.activation = activation
        self.input_dimensions = input_dimensions
        self.lr = lr
        self.weights = weights

        # Initialize neurons
        self.neurons = []

        for neuron_ind_x in range(input_dimensions-kernel_size+1):
            neuron_row = []
            for neuron_ind_y in range(input_dimensions-kernel_size+1):
                neuron_row.append(self.activation, self.num_inputs, lr, weights)
        self.lr = lr

    def calculate(self, inputs: np.ndarray) -> List:
        """
        Calculates the output of the layer.
        :param inputs: Inputs to the layer
        :return: Output of the layer
        """
        outputs = []
        for neuron_layer in self.neurons:
            for neuron in neuron_layer:
                outputs.append(neuron.calculate(inputs))  # Calculate output of each neuron
        return outputs

    def calculate_wdeltas(self, wdeltas_next: List) -> List:
        """
        Calculates the weight deltas of the layer.
        :param wdeltas_next: Weight deltas of the next layer
        :return: Weight deltas of the layer
        """
        wdeltas = []
        for ind, neuron in enumerate(self.neurons):
            # Calculate weight deltas of each neuron
            fwdelta = []
            for fwdeltas in wdeltas_next:
                fwdelta.append(fwdeltas[ind])
            fwdelta = np.sum(fwdelta)
            wdelta = neuron.calc_partial_derivative(fwdelta)
            # Update weights
            neuron.update_weights()
            wdeltas.append(wdelta)
        return wdeltas

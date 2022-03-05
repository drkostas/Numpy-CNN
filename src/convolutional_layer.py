from src import Neuron
import numpy as np
from typing import *


class ConvolutionalLayer:
    def __init__(self, num_kernels: int, kernel_size: int, activation: str, input_dimensions: List, lr: float,
                 weights: np.ndarray):
        """
        Initializes a fully connected layer.
        :param neurons_per_layer: Number of neurons in the layer
        :param activation: Activation function
        :param input_dimensions: Dimensions of input [width/height, channels]
        :param lr: Learning rate
        :param weights: Weights of the layer
        """
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_dimensions = input_dimensions
        self.lr = lr
        self.weights = weights
        self.neurons_per_layer =0
        self.output_size = input_dimensions[0]-kernel_size+1
        # Initialize neurons
        self.kernels = []
        for kernel in range(num_kernels):
            neurons  = []
            for neuron_ind_x in range(input_dimensions[0]-kernel_size+1):
                neuron_row = []
                for neuron_ind_y in range(input_dimensions[0]-kernel_size+1):
                    neuron_row.append(self.activation, self.num_inputs, lr, weights[kernel])
                    self.neurons_per_layer = self.neurons_per_layer+1
                neurons.append(neuron_row)
            self.kernels.append(neurons)

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

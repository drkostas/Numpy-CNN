from src import Neuron
import numpy as np
from typing import *


class ConvolutionalLayer:
    def __init__(self, num_kernels: int, kernel_size: int, input_channels: int,
                 input_dimensions: Sequence[int],
                 activation: str, lr: float, weights: np.ndarray):
        """
        Initializes a fully connected layer.
        :param num_kernels: Number of neurons in the layer
        :param activation: Activation function
        :param input_channels: Number of input channels (input width
        :param input_dimensions: Dimensions of input (height, width)
        :param lr: Learning rate
        :param weights: Weights of the layer
        TODO: If we have time add stride and padding
        """
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.input_dimensions = input_dimensions  # Dimensions of input (height, width)
        self.input_channels = input_channels  # Number of input channels
        self.activation = activation
        self.lr = lr
        self.weights = weights
        self.neurons_per_layer = 0
        self.output_size = input_dimensions[0] - kernel_size + 1
        # Initialize neurons
        self.kernels = []  # Type: List[List[List[Neuron]]]
        for kernel_ind in range(num_kernels):
            neurons = []
            for neuron_ind_x in range(input_dimensions[0] - kernel_size + 1):
                neuron_row = []
                for neuron_ind_y in range(input_dimensions[1] - kernel_size + 1):
                    neuron = Neuron(num_inputs=kernel_size**2,  # TODO:  * input_channels or not?
                                    activation=self.activation,
                                    lr=self.lr,
                                    weights=weights[kernel_ind][neuron_ind_x][neuron_ind_y])
                    neuron_row.append(neuron)
                    self.neurons_per_layer = self.neurons_per_layer + 1
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
        for kernel_ind, kernel in enumerate(self.kernels):
            for kernel_x, kernel_row in enumerate(kernel):
                for kernel_y, neuron in enumerate(kernel_row):
                    print(type(neuron))
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

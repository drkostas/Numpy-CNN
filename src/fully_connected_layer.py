from src import Neuron
import numpy as np
from typing import *


class FullyConnectedLayer:
    def __init__(self, neurons_per_layer: int, activation: str, num_inputs: int, lr: float,
                 weights: np.ndarray):
        """
        Initializes a fully connected layer.
        :param neurons_per_layer: Number of neurons in the layer
        :param activation: Activation function
        :param num_inputs: Number of inputs to each neuron
        :param lr: Learning rate
        :param weights: Weights of the layer
        """
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        self.num_inputs = num_inputs

        # Initialize neurons
        self.neurons = []
        for neuron_ind in range(self.neurons_per_layer):
            self.neurons.append(Neuron(self.activation, self.num_inputs, lr, weights[neuron_ind]))
        self.lr = lr

    def calculate(self, inputs: np.ndarray) -> List:
        """
        Calculates the output of the layer.
        :param inputs: Inputs to the layer
        :return: Output of the layer
        """
        outputs = []
        for neuron in self.neurons:
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

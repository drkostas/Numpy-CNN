from src import FullyConnectedLayer
import numpy as np
from typing import *


class NeuralNetwork:
    def __init__(self, num_inputs: int, loss_function: str, learning_rate: float):
        """
        Initializes a neural network with the given parameters.
        :param num_inputs: The number of inputs to the network.
        :param loss_function: The loss function to use.
        :param learning_rate: The learning rate to use.
        """
        self.num_inputs = num_inputs  # TODO: Make this dynamically inferred.
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        # Initialize the layers
        self.layers = []

    def addLayer(self, num_neurons: int, activation: str, weights: np.ndarray = None):
        """ Adds a layer to the network.
        :param num_neurons: The number of neurons in the new layer.
        :param activation: The activation function for the new layer.
        :param weights: The weights for the new layer.
        :return: None
        """
        num_inputs = self.num_inputs if len(self.layers) == 0 \
            else self.layers[-1].neurons_per_layer
        if weights is None:
            weights = np.random.randn(num_neurons, num_inputs + 1)
        layer = FullyConnectedLayer(num_neurons, activation, num_inputs,
                                    self.learning_rate, weights)
        self.layers.append(layer)

    def calculate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the network for the given inputs.
        :param inputs: The inputs to the network.
        :return: The output of the network.
        """
        for layer in self.layers:
            inputs = layer.calculate(inputs)
        outputs = np.array(inputs)
        return outputs

    def calculate_loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates the loss of the network for the given inputs and targets.
        :param inputs: The inputs to the network.
        :param targets: The targets for the inputs.
        :return: The loss of the network.
        """
        err = 0
        for i in range(len(inputs)):
            outputs = np.array(self.calculate(inputs[i]))
            if self.loss_function == "cross_entropy":
                err = err + self.binary_cross_entropy_loss(outputs, targets[i])
            elif self.loss_function == "square_error":
                err = err + self.square_error_loss(outputs, targets[i])
            else:
                raise ValueError("Invalid loss function.")
        return err

    @staticmethod
    def binary_cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates the binary cross entropy loss of the network for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The cross entropy loss of the network.
        """
        div_by_N = 1 / outputs.shape[0]
        sums = np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
        return div_by_N * (-sums)

    @staticmethod
    def square_error_loss(outputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates the mean squared loss of the network for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The mean squared loss of the network.
        """
        return np.sum((outputs - targets) ** 2) / outputs.shape[0]

    def loss_derivative(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the loss of the network for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The derivative of the loss of the network.
        """
        if self.loss_function == "cross_entropy":
            return self.binary_cross_entropy_loss_derivative(outputs, targets)
        elif self.loss_function == "square_error":
            return self.square_error_loss_derivative(outputs, targets)
        else:
            raise ValueError(f"Invalid loss function. `{self.loss_function}` is not suppoerted.")

    @staticmethod
    def binary_cross_entropy_loss_derivative(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the binary cross entropy loss of the network
        for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The derivative of the cross entropy loss of the network.
        """
        first_term = targets / outputs
        second_term = (1 - targets) / (1 - outputs)
        return -first_term + second_term

    @staticmethod
    def square_error_loss_derivative(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the mean squared loss of the network
        for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The derivative of the mean squared loss of the network.
        """
        return 2 * (np.array(outputs) - np.array(targets)) / np.array(outputs).shape[0]

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Trains the network for the given inputs and targets.
        :param inputs: The inputs to the network.
        :param targets: The targets for the inputs.
        :return: None
        """

        for i in range(len(inputs)):
            # Calculate the outputs of the network for the given input.
            outputs = self.calculate(inputs[i])
            # Calculate the derivative of the loss of the network for the given outputs and targets.
            act_der = [neuron.activation_derivative()
                       for neuron in self.layers[len(self.layers) - 1].neurons]
            wdeltas = [self.loss_derivative(outputs, targets[i]) * act_der]
            # Update the weights of the network.
            for j in range(len(self.layers) - 1, -1, -1):
                wdeltas = self.layers[j].calculate_wdeltas(wdeltas)

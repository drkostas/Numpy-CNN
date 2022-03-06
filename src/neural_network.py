from src import FullyConnectedLayer, ConvolutionalLayer, \
    MaxPoolingLayer, FlattenLayer

import numpy as np
from typing import *


class NeuralNetwork:
    def __init__(self, input_size: Union[int, Tuple], loss_function: str, learning_rate: float,
                 input_channels: int = 1):
        """
        Initializes a neural network with the given parameters.
        :param input_size: The width/height of the input.
        :param loss_function: The loss function to use.
        :param learning_rate: The learning rate to use.
        """
        if isinstance(input_size, int):
            input_size = (input_size,)
        self.input_size = input_size
        self.input_channels = input_channels
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        # Initialize the layers
        self.layers = []

    def addFCLayer(self, num_neurons: int, activation: str, weights: np.ndarray = None):
        """ Adds a layer to the network.
        :param num_neurons: The number of neurons in the new layer.
        :param activation: The activation function for the new layer.
        :param weights: The weights for the new layer.
        :return: None
        """
        # Error checking
        if len(self.layers) == 0:
            if len(self.input_size) == 1:  # 1 dimension
                num_inputs = self.input_size[0]
            else:
                raise ValueError(f"Invalid input size when first layer is FC: "
                                 f"{self.input_size}")
            if self.input_channels > 1:
                raise ValueError(f"Invalid input channels when first layer is FC: "
                                 f"{self.input_channels}")
        else:
            num_inputs = self.layers[-1].neurons_per_layer
        # Create the layer
        if weights is None:
            weights = np.random.randn(num_neurons, num_inputs + 1)
        layer = FullyConnectedLayer(num_neurons, activation, num_inputs,
                                    self.learning_rate, weights)
        self.layers.append(layer)

    addFullyConnectedLayer = addFCLayer

    def addConvLayer(self, num_kernels: int, kernel_size: int, activation: str,
                     weights: np.ndarray = None):
        """ Adds a layer to the network.
        Stride is always 1 and padding is always valid.
        :param num_kernels: The number of neurons in the new layer.
        :param kernel_size: The size of the kernels
        :param activation: The activation function for the new layer.
        :param weights: The weights for the new layer.
        :return: None
        """
        if len(self.layers) == 0:
            if len(self.input_size) == 2:  # 2 dimensions
                input_height, input_width = self.input_size  # Height x Width
                input_channels = self.input_channels
            else:
                raise ValueError(f"Invalid number of inputs when first layer is Convolutional: "
                                 f"{self.input_size}")
        else:
            if isinstance(self.layers[-1], FullyConnectedLayer) \
                    or isinstance(self.layers[-1], FlattenLayer):
                raise ValueError(f"Invalid previous layer type when adding conv layer: "
                                 f"{self.layers[-1]}")
            input_height, input_width = self.layers[-1].output_size
            input_channels = self.layers[-1].output_channels
        if weights is None:
            weights = np.random.randn(num_kernels, kernel_size ** 2 + 1)
        layer = ConvolutionalLayer(num_kernels=num_kernels, kernel_size=kernel_size,
                                   input_channels=input_channels,
                                   input_dimensions=(input_height, input_width),
                                   activation=activation,
                                   lr=self.learning_rate, weights=weights)
        self.layers.append(layer)

    def addMaxPoolLayer(self, kernel_size: int):
        """ Adds a Max Pooling layer to the network.
        :return: None
        """
        if len(self.layers) == 0:
            if len(self.input_size) == 2:  # 2 dimensions
                input_height, input_width = self.input_size  # Height x Width
                input_channels = self.input_channels
            else:
                raise ValueError(f"Invalid number of inputs when first layer is Convolutional: "
                                 f"{self.input_size}")
        else:
            if isinstance(self.layers[-1], FullyConnectedLayer) \
                    or isinstance(self.layers[-1], FlattenLayer):
                raise ValueError(f"Invalid previous layer type when adding conv layer: "
                                 f"{self.layers[-1]}")
            input_height, input_width = self.layers[-1].output_size
            input_channels = self.layers[-1].output_channels
        layer = MaxPoolingLayer(kernel_size=kernel_size,
                                input_channels=input_channels,
                                input_dimensions=(input_height, input_width))
        self.layers.append(layer)

    def addFlattenLayer(self):
        """ Adds a flatten layer to the network.
        :return: None
        """
        if len(self.layers) == 0:
            if len(self.input_size) == 2:  # TODO: check this
                input_height, input_width = self.input_size  # Height x Width
            elif len(self.input_size) == 1:
                input_height, input_width = self.input_size, 1
            else:
                raise ValueError(f"Invalid number of inputs when first layer is Flatten Layer: "
                                 f"{self.input_size}")
            input_channels = self.input_channels
        else:
            input_height, input_width = self.layers[-1].output_size
            input_channels = self.layers[-1].num_kernels
        layer = FlattenLayer(num_inputs=input_height * input_width * input_channels)
        self.layers.append(layer)

    def calculate(self, inputs: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Calculates the output of the network for the given inputs.
        :param inputs: The inputs to the network.
        :return: The output of the network.
        """
        # Error checking
        if self.input_channels > 1:
            if inputs.shape != (self.input_channels, *self.input_size):
                raise ValueError(f"Inputs must be of shape {self.input_channels} x {self.input_size} "
                                 f"but was {inputs.shape}")
        else:
            if inputs.shape != (*self.input_size,):
                raise ValueError(f"Inputs must be of shape {self.input_size} but was {inputs.shape}")
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

        # Calculate
        for layer_ind, layer in enumerate(self.layers):
            print(f"# --- Layer {layer_ind + 1}: {layer.name} --- #")
            print(f"Input shape: {inputs.shape}")
            inputs = layer.calculate(inputs)
            print(f"Output shape: {inputs.shape}")
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

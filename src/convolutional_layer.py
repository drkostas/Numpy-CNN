from src import Neuron
import numpy as np
from typing import *


class ConvolutionalLayer:
    def __init__(self, num_kernels: int, kernel_size: int, input_channels: int,
                 input_dimensions: Sequence[int],
                 activation: str, lr: float, weights: np.ndarray):
        """
        Initializes a convolutional layer.
        :param num_kernels: Number of neurons in the layer
        :param activation: Activation function
        :param input_channels: Number of input channels (input width
        :param input_dimensions: Dimensions of input (height, width)
        :param lr: Learning rate
        :param weights: Weights of the layer
        """
        self.name = "ConvolutionalLayer"
        self.num_kernels = num_kernels
        self.output_channels = num_kernels
        self.kernel_size = kernel_size
        self.input_dimensions = input_dimensions  # Dimensions of input (height, width)
        self.input_channels = input_channels  # Number of input channels
        self.activation = activation
        self.lr = lr
        self.weights = weights
        self.neurons_per_layer = 0
        self.output_size = (input_dimensions[0] - kernel_size + 1,
                            input_dimensions[0] - kernel_size + 1)
        # Initialize neurons
        self.kernels = []  # Type: List[List[List[Neuron]]]
        for kernel_ind in range(num_kernels):
            neurons = []
            for neuron_ind_x in range(input_dimensions[0] - kernel_size + 1):
                neuron_row = []
                for neuron_ind_y in range(input_dimensions[1] - kernel_size + 1):
                    neuron_weights = weights[kernel_ind, :]
                    neuron = Neuron(num_inputs=kernel_size ** 2 * input_channels,
                                    activation=self.activation,
                                    lr=self.lr,
                                    weights=neuron_weights,
                                    input_channels=input_channels)
                    neuron_row.append(neuron)
                    self.neurons_per_layer = self.neurons_per_layer + 1
                neurons.append(neuron_row)
            self.kernels.append(neurons)
        self.neurons = np.array(self.kernels).flatten().tolist()
        self.lr = lr

    def calculate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the layer.
        :param inputs: Inputs to the layer
        :return: Output of the layer
        """
        outputs = []
        for kernel_ind, kernel in enumerate(self.kernels):
            kernel_output = []
            for kernel_x, kernel_row in enumerate(kernel):
                kernel_x_output = []
                for kernel_y, neuron in enumerate(kernel_row):
                    if self.input_channels == 1:  # Shape is (kernel_size x kernel_size)
                        inputs_to_neuron = inputs[kernel_x:kernel_x + self.kernel_size,
                                           kernel_y:kernel_y + self.kernel_size].reshape(-1)
                    else:  # Shape is (num_channels, kernel_size x kernel_size)
                        inputs_to_neuron = inputs[:,
                                           kernel_x:kernel_x + self.kernel_size,
                                           kernel_y:kernel_y + self.kernel_size] \
                            .reshape((self.input_channels, -1))
                    # Calculate output of each neuron
                    kernel_x_output.append(neuron.calculate(inputs_to_neuron))
                kernel_output.append(kernel_x_output)
            outputs.append(kernel_output)
        return np.array(outputs)

    def calculate_wdeltas(self, wdeltas_next: np.ndarray) -> np.ndarray:
        """
        Calculates the weight deltas of the layer.
        :param wdeltas_next: Weight deltas of the next layer
        :return: Weight deltas of the layer
        """
        #wdeltas_next = wdeltas_next[0]
        wdeltas = np.zeros((self.input_channels, *self.input_dimensions))
        for delta_ind_x in range(self.input_dimensions[0]):
            for delta_ind_y in range(self.input_dimensions[1]):
                for kernel_ind in range(self.num_kernels):
                    for delta_next_ind_x in range(self.input_dimensions[0] - self.kernel_size + 1):
                        for delta_next_ind_y in range(self.input_dimensions[1] - self.kernel_size + 1):
                            curr_wdeltas_next = wdeltas_next[kernel_ind,
                                                             delta_next_ind_x,
                                                             delta_next_ind_y]
                            neuron = self.kernels[kernel_ind][delta_next_ind_x][delta_next_ind_y]
                            neuron.derivative(curr_wdeltas_next)
                            neuron.update_weights()
                            neuron_weights = neuron.weights
                            neuron_activ_deriv = neuron.activation_derivative()

                            current_wdeltas = curr_wdeltas_next * neuron_activ_deriv * neuron_weights
                            for input_channel in range(self.input_channels):
                                wdeltas[input_channel,
                                        delta_ind_x,
                                        delta_ind_y] += np.sum(current_wdeltas)
        return wdeltas

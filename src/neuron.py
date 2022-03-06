import numpy as np


class Neuron:
    def __init__(self, activation: str, num_inputs: int,
                 lr: float, weights: np.ndarray, input_channels: int = 1):
        """ Initializes the neuron with the given parameters.
        Args:
            activation: The activation function to use.
            num_inputs: The number of inputs to the neuron.
            lr: The learning rate to use.
            weights: The weights to use.
            input_channels: The number of input channels.
        """
        # Error checking
        if num_inputs/input_channels != (weights.size-1):  # -1 for the bias
            print(f"num_inputs: {num_inputs}")
            print(f"input_channels: {input_channels}")
            print(f"weights.size: {weights.size}")
            raise ValueError(f"Number of inputs/input_channels ({num_inputs/input_channels}) "
                             f"must be equal to (weights.size-1) ({weights.size-1})")
        # Initializes all input vars
        self.activation = activation
        self.num_inputs = num_inputs
        self.input_channels = input_channels
        self.lr = lr
        self.weights = weights
        # Duplicate the weights for each input channel
        if self.input_channels > 1:
            self.weights = np.concatenate((np.tile(self.weights[:-1], self.input_channels),
                                           self.weights[-1].reshape(-1)))
        # Initialize all other object vars
        self.output = None
        self.inputs = None
        self.net = None
        self.partial_der = None

    def activate(self):
        """ Uses the saved net value and activation function to return the output of the node. """
        if self.activation == "linear":
            self.output = self.net
        elif self.activation == "logistic":
            self.output = 1 / (1 + np.exp(-self.net))
        else:
            raise ValueError(f"Activation function not recognized: {self.activation}")
        return self.output

    # Receives a vector of inputs and determines the nodes output using
    # the stored weights and the activation function
    def calculate(self, inputs: np.ndarray):
        """ Receives a vector of inputs and determines the nodes output using
        the stored weights and the activation function.
        The inputs should have two dimensions if there are multiple channels.
        The first dimension is the channel, the second is the input.
        """
        # Check if the input is the correct size
        if self.input_channels > 1:
            if inputs.shape != (self.input_channels, self.num_inputs/self.input_channels):
                raise ValueError(f"Inputs must be of shape "
                                 f"{self.input_channels}x{self.num_inputs/self.input_channels} "
                                 f"but was {inputs.shape}")
        else:
            if inputs.shape != (self.num_inputs,):
                raise ValueError(f"Inputs must be of shape {self.num_inputs} but was {inputs.shape}")

        # Calculate the net value
        self.inputs = np.append(inputs.copy(), [1])
        self.net = np.sum(self.inputs * self.weights)
        return self.activate()

    def activation_derivative(self):
        """ Returns the derivative of the activation function using
        the previously calculated output. """
        if self.activation == "linear":
            return 1
        elif self.activation == "logistic":
            return self.output * (1 - self.output)

    def derivative(self, delta):
        """ Calculates and saves the partial derivative with respect to the weights. """
        self.partial_der = np.array(self.inputs) * delta

    def calc_partial_derivative(self, deltaw_1):
        """ Calculates and saves the partial derivative with respect to the weights. """
        delta = deltaw_1 * self.activation_derivative()
        self.derivative(delta)
        return delta * self.weights

    def update_weights(self):
        """ Updates the nodes weights using the saved partial derivatives and learning rate. """
        self.weights = self.weights - self.lr * self.partial_der

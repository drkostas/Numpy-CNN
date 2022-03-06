from src import Neuron
import numpy as np
from typing import *


class FlattenLayer:
    def __init__(self, num_inputs: int):
        """
        Initializes a flatten layer.
        :param num_inputs: Number of inputs to each neuron
        """
        self.num_inputs = num_inputs

    @staticmethod
    def calculate(inputs: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the layer.
        :param inputs: Inputs to the layer
        :return: Output of the layer
        """
        print(inputs.shape)
        return np.array(inputs).flatten()

    @staticmethod
    def calculate_wdeltas(wdeltas_next: List) -> List:
        """
        Calculates the weight deltas of the layer.
        :param wdeltas_next: Weight deltas of the next layer
        :return: Weight deltas of the layer
        """
        return wdeltas_next

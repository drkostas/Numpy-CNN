import traceback
import argparse
import numpy as np
from src import NeuralNetwork, generateExample, getTensorExample
from typing import *


def get_args() -> argparse.Namespace:
    """Set-up the argument parser

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description='Project 2 for the Deep Learning class (COSC 525). '
                    'Involves the development of a Convolutional Neural Network.',
        add_help=False)
    # Required Args
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-d', '--dataset', required=True,
                               help="The datasets to train the network on. "
                                    "Options: [example1, example2, example3]")
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def main():
    """This is the main function of main.py

    Example:
        python main.py --dataset example1
    """

    # Initializing
    args = get_args()
    # Load the configurations
    dataset_type = args.dataset
    if dataset_type in ('example1', 'example2', 'example3'):
        example_num = int(dataset_type[-1])
        inputs, targets, layers = generateExample(example_num)
        getTensorExample(example_num)
    else:
        raise ValueError('Invalid dataset type')

    # ------- Start of Code ------- #
    # # Initialize the network # #
    netWork = NeuralNetwork(input_size=inputs.shape, loss_function="square_error",
                            learning_rate=100, input_channels=1)
    # Add layers
    for layer in layers:
        if layer['type'] == 'Conv':
            weights = []
            for k_ind in range(layer['num_kernels']):
                kernels = [k_w.flatten() for k_w in layer['weights'][k_ind]]
                kernel_weights = np.concatenate((*kernels,
                                                 layer['biases'][k_ind]))
                weights.append(kernel_weights)
            weights = np.array(weights)
            netWork.addConvLayer(num_kernels=layer['num_kernels'],
                                 kernel_size=layer['kernel_size'],
                                 activation=layer['activation'],
                                 weights=weights)
        elif layer['type'] == 'Flat':
            netWork.addFlattenLayer()
        elif layer['type'] == 'MaxPool':
            netWork.addMaxPoolLayer(kernel_size=layer['kernel_size'])
        elif layer['type'] == 'Dense':
            weights = np.array([np.concatenate((layer['weights'].flatten(), layer['bias']))])
            netWork.addFCLayer(num_neurons=targets.shape[0],
                               activation=layer['activation'],
                               weights=weights)
        else:
            raise ValueError(f'Invalid layer type: {layer["type"]}')

    # # Train the network # #
    # First Feed forward
    outputs = netWork.calculate(inputs=inputs)
    print("----------- Custom Model -----------")
    print(f"model output before:\n{outputs}")

    # Calculate Loss derivative
    loss_der = netWork.loss_derivative(outputs, targets)
    loss = netWork.calculate_loss(np.array([inputs]), targets)
    netWork.train(np.array([inputs]), targets)  # Train the network

    outputs = netWork.calculate(inputs=inputs)
    print(f"model output after: \n{outputs}")

    if example_num == 1:
        print('1st convolutional layer, kernel weights:')
        print(netWork.layers[0].kernels[0][0][0].weights[:-1].reshape((3, 3)))
        print('1st convolutional layer, kernel bias:')
        print(np.array([netWork.layers[0].kernels[0][0][0].weights[-1]]))
        print('fully connected layer weights:')

        print(netWork.layers[2].neurons[0].weights[:-1])
        print('fully connected layer bias:')
        print(np.array([netWork.layers[2].neurons[0].weights[-1]]))
    elif example_num == 2:
        print('1st convolutional layer, 1st kernel weights:')
        print(netWork.layers[0].kernels[0][0][0].weights[:-1].reshape((3, 3)))
        print('1st convolutional layer, 1st kernel bias:')
        print(np.array([netWork.layers[0].kernels[0][0][0].weights[-1]]))

        print('1st convolutional layer, 2st kernel weights:')
        print(netWork.layers[0].kernels[1][0][0].weights[:-1].reshape((3, 3)))
        print('1st convolutional layer, 2st kernel bias:')
        print(np.array([netWork.layers[0].kernels[1][0][0].weights[-1]]))

        print('2nd convolutional layer, 1st kernel weights:')
        print(netWork.layers[1].kernels[0][0][0].weights[:-1].reshape((2, 3, 3)))
        print('2nd convolutional layer, 1st kernel bias:')
        print(np.array([netWork.layers[1].kernels[0][0][0].weights[-1]]))

        print('fully connected layer weights:')

        print(netWork.layers[3].neurons[0].weights[:-1])
        print('fully connected layer bias:')
        print(np.array([netWork.layers[3].neurons[0].weights[-1]]))
    elif example_num == 3:
        print('1st convolutional layer, 1st kernel weights:')
        print(netWork.layers[0].kernels[0][0][0].weights[:-1].reshape((3, 3)))
        print('1st convolutional layer, 1st kernel bias:')
        print(np.array([netWork.layers[0].kernels[0][0][0].weights[-1]]))

        print('1st convolutional layer, 2st kernel weights:')
        print(netWork.layers[0].kernels[1][0][0].weights[:-1].reshape((3, 3)))
        print('1st convolutional layer, 2st kernel bias:')
        print(np.array([netWork.layers[0].kernels[1][0][0].weights[-1]]))

        print('fully connected layer weights:')

        print(netWork.layers[3].neurons[0].weights[:-1])
        print('fully connected layer bias:')
        print(np.array([netWork.layers[3].neurons[0].weights[-1]]))
    else:
        raise ValueError(f'Invalid example number: {example_num}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e

# # First Layer (Convolutional)
#    weights_L1 = np.array(
#        [np.concatenate((l1k1.flatten(), l1b1)), np.concatenate((l1k2.flatten(), l1b2))])
#    netWork.addConvLayer(num_kernels=2, kernel_size=3, activation="logistic", weights=weights_L1)
#    # Second Layer (Convolutional)
#    weights_L2 = np.array([np.concatenate((l2c1.flatten(), l2c2.flatten(), l2b))])
#    netWork.addConvLayer(num_kernels=1, kernel_size=3, activation="logistic", weights=weights_L2)
#    # Third Layer (Fully Connected)
#    netWork.addFlattenLayer()
#    weights_L3 = np.array([np.concatenate((l3.flatten(), l3b))])
#    netWork.addFCLayer(num_neurons=1, activation="logistic", weights=weights_L3)

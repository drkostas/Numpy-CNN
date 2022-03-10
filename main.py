import traceback
import argparse
import numpy as np
from src import NeuralNetwork, \
    generateExample1, generateExample2, generateExample3,\
    tensorExample1, tensorExample2, tensorExample3

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
                                    "Options: [and, xor, class_example]")
    required_args.add_argument('-n', '--network', required=True,
                               help="The network configuration to use. "
                                    "Options: [1x1_net, 2x1_net, 2x2_net]")
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def get_network_config(network_name: str) -> Dict[str, Any]:
    """Get the network configuration

    Args:
        network_name (str): The name of the network to get the configuration for

    Returns:
        Dict[str, Any]: The network configuration
    """
    nn_conf = {}
    if network_name == '1x1_net':
        nn_conf['neurons_per_layer'] = [1]
        nn_conf['activations'] = ['logistic']
        nn_conf['loss_function'] = 'square_error'
        nn_conf['learning_rate'] = 5
        nn_conf['epochs'] = 5000
        nn_conf['print_every'] = 500
    elif network_name == '2x1_net':
        nn_conf['neurons_per_layer'] = [2, 1]
        nn_conf['activations'] = ['logistic', 'logistic']
        nn_conf['loss_function'] = 'square_error'
        nn_conf['learning_rate'] = 5
        nn_conf['epochs'] = 5000
        nn_conf['print_every'] = 500
    elif network_name == '2x2_net':
        nn_conf['neurons_per_layer'] = [2, 2]
        nn_conf['activations'] = ['logistic', 'logistic']
        nn_conf['loss_function'] = 'cross_entropy'
        nn_conf['learning_rate'] = 0.5
        nn_conf['epochs'] = 100
        nn_conf['print_every'] = 100
    else:
        raise ValueError(f"Network name {network_name} not recognized.")

    return nn_conf


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get the dataset configuration

    Args:
        dataset_name (str): The name of the dataset to get the configuration for

    Returns:
        Dict[str, Any]: The dataset configuration
    """
    dataset_conf = {}
    if dataset_name == 'example1':
        dataset_conf['inputs'] = []
        dataset_conf['outputs'] = []  # TODO: change me
    if dataset_name == 'and':
        dataset_conf['inputs'] = [[0, 0], [0, 1], [1, 0], [1, 1]]
        dataset_conf['outputs'] = [[0], [0], [0], [1]]
    elif dataset_name == 'xor':
        dataset_conf['inputs'] = [[0, 0], [0, 1], [1, 0], [1, 1]]
        dataset_conf['outputs'] = [[0], [1], [1], [0]]
    elif dataset_name == 'class_example':
        dataset_conf['inputs'] = [0.05, 0.1]
        dataset_conf['desired_outputs'] = [0.01, 0.99]
        dataset_conf['weights'] = [[[0.15, 0.20, 0.35], [0.25, 0.30, 0.35]],
                                   [[0.40, 0.45, 0.60], [0.50, 0.55, 0.60]]]
    else:
        raise ValueError(f"Dataset name {dataset_name} not recognized.")

    return dataset_conf


def main() -> argparse.Namespace:
    """This is the main function of main.py

    Example:
        python main.py --dataset xor --network 2x1_net
    """

    # Initializing
    # args = get_args()
    # Load the configurations
    # nn_type = args.network
    # nn_conf = get_network_config(nn_type)
    # dataset_type = args.dataset
    # dataset_conf = get_dataset_config(dataset_type)

    # netWork = NeuralNetwork(input_size=10, loss_function="cross_entropy",
    #                         learning_rate=.1, input_channels=1)
    # netWork.addFCLayer(num_neurons=5, activation="logistic")
    # netWork.addFCLayer(num_neurons=2, activation="logistic")
    # tin = np.arange(1, 201).reshape(20, 10)
    # tout = np.array([[7, 8]])
    # netWork.train(tin, tout)
    # import sys
    # sys.exit()

    # ------- Start of Code ------- #
    example = 3
    if (example == 1):
        tensorExample1()
        l1, l1b, l2, l2b, input, targets = generateExample1()
        weights_L1 = np.array([np.concatenate((l1.flatten(), l1b))])
        netWork = NeuralNetwork(input_size=(5, 5), loss_function="square_error",
                                learning_rate=100, input_channels=1)
        netWork.addConvLayer(num_kernels=1, kernel_size=3, activation="logistic", weights=weights_L1)
        netWork.addFlattenLayer()
        weights_L2 = np.array([np.concatenate((l2.flatten(), l2b))])
        netWork.addFCLayer(num_neurons=1, activation="logistic", weights=weights_L2)
        outputs = netWork.calculate(inputs=input)

        print("----------- Custom Model -----------")
        print(f"model output before:\n{outputs}")

        # Calculate Loss derivative
        loss_der = netWork.loss_derivative(outputs, targets)
        loss = netWork.calculate_loss([input], targets)
        netWork.train(np.array([input]), targets)  # Train the network

        outputs = netWork.calculate(inputs=input)
        print(f"model output after: \n{outputs}")

        print('1st convolutional layer, kernel weights:')
        print(netWork.layers[0].kernels[0][0][0].weights[:-1].reshape((3,3)))
        print('1st convolutional layer, kernel bias:')
        print(np.array([netWork.layers[0].kernels[0][0][0].weights[-1]]))
        print('fully connected layer weights:')

        print(netWork.layers[2].neurons[0].weights[:-1])
        print('fully connected layer bias:')
        print(np.array([netWork.layers[2].neurons[0].weights[-1]]))

    elif(example==2):
        tensorExample2()
        l1k1, l1k2, l1b1, l1b2, l2c1, l2c2, l2b, l3, l3b, input, targets = generateExample2()
        weights_L1 = np.array([np.concatenate((l1k1.flatten(),l1b1)),np.concatenate((l1k2.flatten(),l1b2))])
        netWork = NeuralNetwork(input_size=(7, 7), loss_function="square_error",
                                learning_rate=100, input_channels=1)
        netWork.addConvLayer(num_kernels=2, kernel_size=3, activation="logistic",weights=weights_L1)
        weights_L2 = np.array([np.concatenate((l2c1.flatten(),l2c2.flatten(), l2b))])
        netWork.addConvLayer(num_kernels=1, kernel_size=3, activation="logistic",weights=weights_L2)
        netWork.addFlattenLayer()
        weights_L3 = np.array([np.concatenate((l3.flatten(),l3b))])
        netWork.addFCLayer(num_neurons=1, activation="logistic",weights=weights_L3)
        outputs = netWork.calculate(inputs=input)
        print("----------- Custom Model -----------")
        print(f"model output before:\n{outputs}")

        # Calculate Loss derivative
        loss_der = netWork.loss_derivative(outputs, targets)
        loss = netWork.calculate_loss([input], targets)
        netWork.train(np.array([input]), targets)  # Train the network

        outputs = netWork.calculate(inputs=input)
        print(f"model output after: \n{outputs}")

        print('1st convolutional layer, 1st kernel weights:')
        print(netWork.layers[0].kernels[0][0][0].weights[:-1].reshape((3, 3)))
        print('1st convolutional layer, 1st kernel bias:')
        print(np.array([netWork.layers[0].kernels[0][0][0].weights[-1]]))

        print('1st convolutional layer, 2st kernel weights:')
        print(netWork.layers[0].kernels[1][0][0].weights[:-1].reshape((3, 3)))
        print('1st convolutional layer, 2st kernel bias:')
        print(np.array([netWork.layers[0].kernels[1][0][0].weights[-1]]))

        print('2nd convolutional layer, 1st kernel weights:')
        print(netWork.layers[1].kernels[0][0][0].weights[:-1].reshape((2,3, 3)))
        print('2nd convolutional layer, 1st kernel bias:')
        print(np.array([netWork.layers[1].kernels[0][0][0].weights[-1]]))

        print('fully connected layer weights:')

        print(netWork.layers[3].neurons[0].weights[:-1])
        print('fully connected layer bias:')
        print(np.array([netWork.layers[3].neurons[0].weights[-1]]))

    elif (example == 3):
        tensorExample3()
        l1k1, l1k2, l1b1, l1b2, l3, l3b, input, targets = generateExample3()
        weights_L1 = np.array([np.concatenate((l1k1.flatten(), l1b1)), np.concatenate((l1k2.flatten(), l1b2))])
        netWork = NeuralNetwork(input_size=(8, 8), loss_function="square_error",
                                learning_rate=100, input_channels=1)
        netWork.addConvLayer(num_kernels=2, kernel_size=3, activation="logistic", weights=weights_L1)
        netWork.addMaxPoolLayer(kernel_size=2)
        netWork.addFlattenLayer()
        weights_L3 = np.array([np.concatenate((l3.flatten(), l3b))])
        netWork.addFCLayer(num_neurons=1, activation="logistic", weights=weights_L3)
        outputs = netWork.calculate(inputs=input)
        print("----------- Custom Model -----------")
        print(f"model output before:\n{outputs}")

        # Calculate Loss derivative
        loss_der = netWork.loss_derivative(outputs, targets)
        loss = netWork.calculate_loss([input], targets)
        netWork.train(np.array([input]), targets)  # Train the network

        outputs = netWork.calculate(inputs=input)
        print(f"model output after: \n{outputs}")

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

    """
    print(f'Training the `{nn_type}` network on the `{dataset_type}` dataset.')
    # Train the network
    inputs = np.array(dataset_conf['inputs'])
    outputs = np.array(dataset_conf['outputs'])
    # Initialize the network
    # Initialize the network
    netWork = NeuralNetwork(num_inputs=inputs.shape[1],
                            loss_function=nn_conf['loss_function'],
                            learning_rate=nn_conf['learning_rate'])
    # Add the layers
    for num_neurons, activation in zip(nn_conf['neurons_per_layer'], nn_conf['activations']):
        netWork.addLayer(num_neurons=num_neurons, activation=activation)
    # Train the network for the given number of epochs
    for epoch in range(nn_conf['epochs']):
        netWork.train(inputs, outputs)  # Train the network
        loss = netWork.calculate_loss(inputs, outputs)  # Calculate the loss
        if epoch % nn_conf['print_every'] == 0:
            print(f"Epoch: {epoch} Loss: {loss}")
    print(f"Epoch: {nn_conf['epochs']} Loss: {loss}")
    # Test on the predictions
    print(f'Predictions on the {dataset_type} dataset')
    for inp, outp in zip(inputs, outputs):
        print(f"True Output: {outp} Prediction: {netWork.calculate(inp)[0]}")
        """


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e

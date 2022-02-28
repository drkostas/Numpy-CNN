import traceback
import argparse
import numpy as np
from src import Configuration, ColorLogger, Neuron, FullyConnectedLayer, NeuralNetwork
from typing import *

# Color-logger is used to print colored messages to the console
logger = ColorLogger(logger_name='Main', color='yellow')


def get_args() -> argparse.Namespace:
    """Set-up the argument parser

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description='Project 1 for the Deep Learning class (COSC 525). '
                    'Involves the development of a FeedForward Neural Network.',
        add_help=False)
    # Required Args
    required_args = parser.add_argument_group('Required Arguments')
    config_file_params = {
        'type': argparse.FileType('r'),
        'required': True,
        'help': "The path to the yaml configuration file."
    }
    required_args.add_argument('-d', '--dataset', required=True,
                               help="The datasets to train the network on. "
                                    "Options (defined in yml): [and, xor, class_example]")
    required_args.add_argument('-n', '--network', required=True,
                               help="The network configuration to use. "
                                    "Options (defined in yml): [1x1_net, 2x1_net, 2x2_net]")
    required_args.add_argument('-c', '--config-file', **config_file_params)
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument('-l', '--log', required=False, default='out.log',
                               help="Name of the output log file")
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def main():
    """This is the main function of main.py

    Example:
        python main.py --dataset xor --network 2x1_net --config confs/main_conf.yml
    """

    # Initializing
    get_conf = lambda c_type, c: next(filter(lambda x: x['type'] == c_type, c))
    args = get_args()
    ColorLogger.setup_logger(log_path=args.log, clear_log=True)
    # Load the configurations
    config = Configuration(config_src=args.config_file)
    nn_structures = config.get_config('neural-network')
    nn_structure = get_conf(args.network, nn_structures)
    nn_conf = nn_structure['config']
    nn_type = nn_structure['type']
    datasets = config.get_config('dataset')
    dataset = get_conf(args.dataset, datasets)
    dataset_conf = dataset['config']
    dataset_type = dataset['type']

    # ------- Start of Code ------- #
    logger.nl()  # New line
    logger.info(f'Training the `{nn_type}` network on the `{dataset_type}` dataset.', color='green')
    if args.dataset != 'class_example':  # XOR and AND cases
        # Train the network
        inputs = np.array(dataset_conf['inputs'])
        outputs = np.array(dataset_conf['outputs'])
        # Initialize the network
        netWork = NeuralNetwork(num_layers=len(nn_conf['neurons_per_layer']),
                                neurons_per_layer=nn_conf['neurons_per_layer'],
                                activations=nn_conf['activations'],
                                num_inputs=inputs.shape[1],
                                loss_function=nn_conf['loss_function'],
                                learning_rate=nn_conf['learning_rate'])
        # Train the network for the given number of epochs
        for epoch in range(nn_conf['epochs']):
            netWork.train(inputs, outputs)  # Train the network
            loss = netWork.calculate_loss(inputs, outputs)  # Calculate the loss
            if epoch % nn_conf['print_every'] == 0:
                logger.info(f"Epoch: {epoch} Loss: {loss}")
        logger.info(f"Epoch: {nn_conf['epochs']} Loss: {loss}")
        # Test on the predictions
        logger.info(f'Predictions on the {dataset_type} dataset', color='cyan')
        for inp, outp in zip(inputs, outputs):
            logger.info(f"True Output: {outp} Prediction: {netWork.calculate(inp)[0]}", color='cyan')
    else:  # Class Example
        # Set up the weights and biases based on the class example
        inputs = [np.array(dataset_conf['inputs'])]
        desired_outputs = np.array(dataset_conf['desired_outputs'])
        weights = [np.array(weight) for weight in dataset_conf['weights']]
        # Initialize the network using the predefined weights and biases
        netWork = NeuralNetwork(num_layers=len(nn_conf['neurons_per_layer']),
                                neurons_per_layer=nn_conf['neurons_per_layer'],
                                activations=nn_conf['activations'],
                                num_inputs=2,
                                loss_function=nn_conf['loss_function'],
                                learning_rate=nn_conf['learning_rate'],
                                weights=weights)  # Give the network the weights and biases
        # Print the network inputs and weights before training
        logger.info("Pre-training Inputs:")
        logger.info(f"{inputs[0]}", color="cyan")
        logger.info("Pre-training Weights:")
        logger.info(
            f"{netWork.layers[0].neurons[0].weights} (h1) x {netWork.layers[1].neurons[0].weights} (O1)",
            color="cyan")
        logger.info(
            f"{netWork.layers[0].neurons[1].weights} (h1) x {netWork.layers[1].neurons[1].weights} (O1)",
            color="cyan")
        # Activate the network
        outputs = netWork.calculate(inputs[0])  # Feed-forward the network
        logger.info(f"Outputs after calling `activate()`:")
        logger.info(f"{outputs}", color="cyan")
        # Calculate the wdeltas - single step of backpropagation
        wdeltas = [netWork.loss_derivative(np.array(outputs), desired_outputs)]
        for j in range(len(netWork.layers) - 1, -1, -1):
            wdeltas = netWork.layers[j].calculate_wdeltas(wdeltas)
        # Print the wdeltas, the weights, and the outputs after backpropagation
        logger.info("Wdeltas after calling `calculate_wdeltas()`:")
        logger.info(f"{wdeltas}", color="cyan")
        logger.info("Weights after a single step of back-propagation:")
        logger.info(
            f"{netWork.layers[0].neurons[0].weights} (h1) x {netWork.layers[1].neurons[0].weights} (O1)",
            color="cyan")
        logger.info(
            f"{netWork.layers[0].neurons[1].weights} (h1) x {netWork.layers[1].neurons[1].weights} (O1)",
            color="cyan")
        outputs = netWork.calculate(inputs[0])
        logger.info("Post-training Outputs:")
        logger.info(f"{outputs}", color="cyan")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e

import numpy as np


# Generate data and weights for "example2"
def generateExample(num: int, seed: int = 10) -> (np.ndarray, np.ndarray):
    # Set a seed (that way you get the same values when rerunning the function)
    np.random.seed(seed)

    if num == 1:
        # First hidden layer, one kernel
        l1k1 = np.random.rand(3, 3)
        l1b1 = np.random.rand(1)
        # output layer, fully connected
        l2 = np.random.rand(1, 9)
        l2b = np.random.rand(1)
        # Input and Output
        inputs = np.random.rand(5, 5)
        output = np.random.rand(1)
        # Construct Dict with info about the layers
        layers = ({'type': 'Conv',  # First layer
                   'num_kernels': 1,
                   'kernel_size': 3,
                   'weights': [[l1k1]],
                   'biases': [l1b1],
                   'activation': 'logistic'},
                  {'type': 'Flat'},  # Second layer
                  {'type': 'Dense',  # Third layer
                   'weights': l2,
                   'bias': l2b,
                   'activation': 'logistic'})
    elif num == 2:
        # First hidden layer, two kernels
        l1k1 = np.random.rand(3, 3)
        l1k2 = np.random.rand(3, 3)
        l1b1 = np.random.rand(1)
        l1b2 = np.random.rand(1)

        # second hidden layer, one kernel, two channels
        l2c1 = np.random.rand(3, 3)
        l2c2 = np.random.rand(3, 3)
        l2b = np.random.rand(1)

        # output layer, fully connected
        l3 = np.random.rand(1, 9)
        l3b = np.random.rand(1)

        # input and output
        inputs = np.random.rand(7, 7)
        output = np.random.rand(1)
        # Construct Dict with info about the layers
        layers = ({'type': 'Conv',  # First layer
                   'num_kernels': 2,
                   'kernel_size': 3,
                   'weights': [[l1k1], [l1k2]],
                   'biases': [l1b1, l1b2],
                   'activation': 'logistic'},
                  {'type': 'Conv',  # Second layer
                   'num_kernels': 1,
                   'kernel_size': 3,
                   'weights': [[l2c1, l2c2]],
                   'biases': [l2b],
                   'activation': 'logistic'},
                  {'type': 'Flat'},  # Third layer
                  {'type': 'Dense',  # Third layer
                   'weights': l3,
                   'bias': l3b,
                   'activation': 'logistic'})
    elif num == 3:
        # First hidden layer, two 3x3 kernels
        l1k1 = np.random.rand(3, 3)
        l1k2 = np.random.rand(3, 3)
        l1b1 = np.random.rand(1)
        l1b2 = np.random.rand(1)

        # second hidden layer, 2x2 max pooling

        # output layer, fully connected
        l3 = np.random.rand(1, 50)
        l3b = np.random.rand(1)

        # input and output
        inputs = np.random.rand(8, 8)
        output = np.random.rand(1)
        # Construct Dict with info about the layers
        layers = ({'type': 'Conv',  # First layer
                   'num_kernels': 2,
                   'kernel_size': 3,
                   'weights': [[l1k1], [l1k2]],
                   'biases': [l1b1, l1b2],
                   'activation': 'logistic'},
                  {'type': 'MaxPool',  # Second layer
                   'kernel_size': 2},
                  {'type': 'Flat'},  # Third layer
                  {'type': 'Dense',  # Third layer
                   'weights': l3,
                   'bias': l3b,
                   'activation': 'logistic'})
    else:
        raise ValueError("Example number must be 1, 2 or 3")

    return inputs, output, layers

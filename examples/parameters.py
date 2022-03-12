import numpy as np

# Generate data and weights for "example2"
def generateExample1():
    # Set a seed (that way you get the same values when rerunning the function)
    np.random.seed(10)

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
    input = np.random.rand(5, 5)
    output = np.random.rand(1)

    return l1k1, l1k2, l1b1, l1b2, l2c1, l2c2, l2b, l3, l3b, input, output

# Generate data and weights for "example2"
def generateExample2():
    # Set a seed (that way you get the same values when rerunning the function)
    np.random.seed(10)

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
    input = np.random.rand(7, 7)
    output = np.random.rand(1)

    return l1k1, l1k2, l1b1, l1b2, l2c1, l2c2, l2b, l3, l3b, input, output

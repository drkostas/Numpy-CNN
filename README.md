# COSC525: Project 2: Convolutional Neural Networks

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/drkostas/COSC525-Project2/blob/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
+ [Installing the requirements](#installing)
+ [Running the code](#run_locally)
    + [Execution Options](#execution_options)
        + [main.py](#src_main)
+ [Todo](#todo)
+ [License](#license)

## About <a name = "about"></a>

Project 2 for the Deep Learning course (COSC 525). Involves the development of a 
Convolutional Neural Network.

The main code is located in the [main.py](main.py) file. The Neuron, FullyConnectedLayer, 
and NeuralNetwork classes are located in the [src folder](src).

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python > 3.6 and any Bash based shell (e.g. zsh) installed.

```ShellSession
$ python3.8 -V
Python 3.8.2

$ echo $SHELL
/usr/bin/zsh
```

## Installing the requirements <a name = "installing"></a>

All the installation steps are being handled by the [Makefile](Makefile). You can either use conda or
venv by setting the flag `env=<conda|venv>`. To load an env file use the
flag `env_file=<path to env file>`

Before installing everything, make any changes needed in the [settings.ini](settings.ini) file.

Then, to create a conda environment, install the requirements, setup the library and run the tests
execute the following command:

```ShellSession
$ make install
```

## Running the code <a name = "run_locally"></a>

In order to run the code, you will only need to change the yml file if you need to, and either run its
file directly or invoke its console script.

### Execution Options <a name = "execution_options"></a>

First, make sure you are in the correct virtual environment:

```ShellSession
$ conda activate cosc525_project2

$ which python
/home/<user>/anaconda3/envs/src/bin/python
```

#### main.py <a name = "src_main"></a>

Now, in order to run the code you can call the [main.py](main.py)
directly.

```ShellSession
$ python main.py -h
usage: main.py -d DATASET -n NETWORK -c CONFIG_FILE [-l LOG] [-h]

Project 2 for the Deep Learning class (COSC 525). Involves the development of a Convolutional Neural Network.

Required Arguments:
  -d DATASET, --dataset DATASET
                        The datasets to train the network on. Options (defined in yml): [and, xor, class_example]
  -n NETWORK, --network NETWORK
                        The network configuration to use. Options (defined in yml): [1x1_net, 2x1_net, 2x2_net]
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        The path to the yaml configuration file.

Optional Arguments:
  -l LOG, --log LOG     Name of the output log file
  -h, --help            Show this help message and exit
```

## TODO <a name = "todo"></a>

Read the [TODO](TODO.md) to see the current task list.

## License <a name = "license"></a>

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

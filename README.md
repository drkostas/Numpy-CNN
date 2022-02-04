# COSC525: Project 1<img src='https://github.com/drkostas/COSC525-Project1/blob/master/img/snek.png' align='right' width='180' height='104'>

[![CircleCI](https://circleci.com/gh/drkostas/COSC525-Project1/tree/master.svg?style=svg)](
https://circleci.com/gh/drkostas/COSC525-Project1/tree/master)
[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/drkostas/COSC525-Project1/blob/master/LICENSE)

## Table of Contents

+ [About](#about)
+ [Getting Started](#getting_started)
    + [Prerequisites](#prerequisites)
+ [Installing, Testing, Building](#installing)
+ [Running the code](#run_locally)
    + [Execution Options](#execution_options)
        + [COSC525Project1 Main](#src_main)
+ [Todo](#todo)
+ [License](#license)

## About <a name = "about"></a>

Project 1 for the Deep Learning class (COSC 525). Involves the development of a FeedForward Neural
Network.

This project already includes the following packages:

- [yaml-config-wrapper](https://pypi.org/project/yaml-config-wrapper/)
- [termcolor-logger](https://pypi.org/project/termcolor-logger/)

To get started, follow their respective instructions.

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for
development and testing purposes. See deployment for notes on how to deploy the project on a live
system.

### Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python > 3.6 and any Bash based shell (e.g. zsh) installed.

```ShellSession

$ python3.6 -V
Python 3.6

$ echo $SHELL
/usr/bin/zsh

```

## Installing, Testing, Building <a name = "installing"></a>

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
$ conda activate cosc525_project1

$ which python
/home/drkostas/anaconda3/envs/src/bin/python

```

#### main.py <a name = "src_main"></a>

Now, in order to run the code you can call the [main.py](main.py)
directly.

```ShellSession
$ python main.py --help
usage: main.py -c CONFIG_FILE [-l LOG] [-d] [-h]

Project 1 for the Deep Learning class (COSC 525). Involves the development of a FeedForward Neural Network.

Required Arguments:
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        The configuration yml file

Optional Arguments:
  -l LOG, --log LOG     Name of the output log file
  -d, --debug           Enables the debug log messages
  -h, --help            Show this help message and exit
```

## TODO <a name = "todo"></a>

Read the [TODO](TODO.md) to see the current task list.

## License <a name = "license"></a>

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

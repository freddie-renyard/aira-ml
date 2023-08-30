## Overview

In active development.

A repository for the development of a compiler which compiles machine learning models to custom FPGA hardware. 

## Usage

The following command can be used to build networks with Aira:

`python aira_ml.py -f /model/dense_mnist`

where -f is the path to the target model.

## Installation and Usage

Aira requires a working installation of TensorFlow to inspect models, and Vivado to build the FPGA firmware. Other FPGA toolchains can be used to build the SystemVerilog output, but Vivado is currently built in to Aira's end-to-end workflow.

1. Clone this repository and `cd` into the aira-ml directory.

`https://github.com/freddie-renyard/aira-ml.git`
`cd aira-ml`

2. Before building a network, the tools must be configured with the locations of the FPGA project. This can be found in the `aira-ml/aira_ml/config/server_config.json` file. The required fields are listed below:

- `vivado_loc`: The location of the Vivado `settings64.sh` setup file.
- `project_loc`: The location of the Vivado Aira project file, `aira_project.xpr`.
- `bitstream_loc`: The location of the implemented bitstream output. This will be in the `*.runs/impl_1/` directory within the Aira project file.
- `device_name`: The name of the FPGA within Vivado. E.g. for Digilent's Genesys 2 board, this could be the `xc7k325t_0` board.

## Supported Networks

At present Sequential TensorFlow models are supported by Aira. Currently the following TensorFlow layers are supported:

- tf.keras.layers.Dense
- tf.keras.layers.Conv2D
    - At present, these must be followed by a MaxPooling2D layer.
    - Odd-numbered kernel sizes are supported.
    - Valid or same padding can be used.
    - Stride and dilatation rate must be (1, 1). 
- tf.keras.layers.MaxPooling2D
    - At present, these must follow a Conv2D layer.
    - Supported padding: valid
    - Supported kernel size: (2, 2)

- Supported activation functions:
    - ReLU
    - Sigmoid
    - None (y = x)


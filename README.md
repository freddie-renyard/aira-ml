## Overview

In active development.

A repository for the development of a compiler which compiles machine learning models to custom FPGA hardware. 

## Usage

The following command can be used to build networks with Aira:

```bash
python aira_ml.py -f /model/dense_mnist
```

where -f is the path to the target model.

## Installation and Usage

Aira requires a working installation of TensorFlow to inspect models, and Vivado to build the FPGA firmware. Other FPGA toolchains can be used to build the SystemVerilog output, but Vivado is currently built in to Aira's end-to-end workflow.

1. Clone this repository and `cd` into the aira-ml directory.

```bash
https://github.com/freddie-renyard/aira-ml.git
cd aira-ml
```

2. Before building a network, the tools must be configured with the locations of the FPGA project. This can be found in the `aira-ml/aira_ml/config/server_config.json` file. The required fields are listed below:

| Field | Description |
| --- | ----------- |
| `vivado_loc` | The location of the Vivado `settings64.sh` setup file. |
| `project_loc` | The location of the Vivado Aira project file, `aira_project.xpr`. |
| `bitstream_loc` | The location of the implemented bitstream output. This will be in the `*.runs/impl_1/` directory within the Aira project file. |
| `device_name` | The name of the FPGA within Vivado. E.g. for Digilent's Genesys 2 board, this could be the `xc7k325t_0` board. |

3. Use the command above to build a model.

```bash
python aira_ml.py -f /path/to/tensorflow/model/
```

## Modifying threading parameters

Aira can efficiently parallelise different portions of the computation needed to evaluate an inference. This is determined by the threading parameters, which vary between layers. These are described below:

- Dense
    - `threads`: parellises the computation for each neuron.
- Conv2D
    - `filter_threads`: parellises the evaluation of each kernel within a filter.
    - `rowcol_threads`: parellises the evaluation of the input image.
    - Evaluation of the channels of an image is parellised by default. 

Each thread will evaluate an equal proportion of the work of each layer. For instance, in a Dense layer with 64 neurons and 2 threads, each thread will evaluate 32 neurons.

1. Open the `aira-ml/aira_ml/config/compiler_config.json` file.

2. Use the `model.summary()` command within python or a network analysis tool like [Netron](https://netron.app/) to determine the names of each layer, e.g. `conv2d_1, dense_2`.

3. Use the JSON file to specify threads for each layer. This can be done by:

- Specifying an integer number of threads
- Specifying a float, which will use a proportion of the threads that are possible for the layer. 
- Specifying -1, which will synthesise the maximum number of threads.

By default, 1 thread is used.

### Example

For a network with this architecture (`aira-ml/models/conv_mnist/model`):

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 6)         60        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 6)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 8)         440       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 8)           0         
_________________________________________________________________
flatten (Flatten)            (None, 200)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               25728     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 27,518
Trainable params: 27,518
Non-trainable params: 0
_________________________________________________________________
```

The following JSON could be used:

```
"thread_config": {
    "conv2d": {
        "filter_threads": -1,
        "rowcol_threads": 4
    },
    "dense": {
        "threads": 0.5
    },
}
```

This would result in:

- The first layer (`conv2d`) having 6 filter threads and 4 rowcol threads.
- The sixth layer (`dense`) having 64 threads.
- All other layers having 1 thread.

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
- tf.keras.layers.Flatten
    - Must be preceded by Conv2D and followed by Dense.

- Supported activation functions:
    - ReLU
    - Sigmoid
    - None (y = x)


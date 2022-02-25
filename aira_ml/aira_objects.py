import numpy as np
from aira_ml.tools.aira_exceptions import AiraException

class DenseAira:

    def __init__(self, index, weights, biases, act_name):
        
        self.index = index

        # Ensure that the activation function used in the layer has
        # hardware support.
        self.act_name = None
        if act_name == 'relu':
            self.act_ = act_name
        else:
            raise AiraException("Unsupported function found in a Dense layer: {}".format(act_name))

        # Infer the number of neurons from the dimensionality of the weight matrix
        weight_dims = np.shape(weights)
        self.pre_neuron_num = weight_dims[0]
        self.post_neuron_num = weight_dims[1]

        # Check that the weights are stored in a valid way.
        if np.shape(weight_dims)[0] != 2:
            raise AiraException("The weights for Dense layer {} are not stored in a 2D tensor.".format(index))
            
        # Compile the weights and biases.

        
        

import numpy as np
from aira_ml.tools.aira_exceptions import AiraException

class DenseAira:

    def __init__(self, index, weights, biases, act_name):
        
        self.index = index

        self.raw_weights = weights
        self.raw_biases = biases

        # Ensure that the activation function used in the layer has
        # hardware support.
        self.act_name = None
        if act_name == 'relu':
            self.act_ = act_name
        else:
            raise AiraException("Unsupported function found in a Dense layer: {}".format(act_name))
        

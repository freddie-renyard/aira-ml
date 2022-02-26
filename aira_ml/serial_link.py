from lib2to3.pgen2.token import N_TOKENS
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model

class SerialLink:

    def __init__(self):
        pass
    
    def flatten_tensor(self, in_tensor):
        """Flatten an input tensor to allow for serial transmission 
        to the FPGA.
        """
        in_tensor = np.array(in_tensor)

        # TODO Fetch this data from the SerialLink object rather than
        # computing it each time the method is called.
        tensor_dims = np.shape(in_tensor)
        flat_shape = np.prod(tensor_dims)

        return np.reshape(in_tensor, (flat_shape), order="C")

    def trial_flatten(self):
        """Test method to allow testing of different input tensor shapes
        to ensure that they are flattened appropriately.
        """
        inputs = Input(shape=(3,3))
        prediction = Flatten()(inputs)
        model = Model(inputs=inputs, outputs=prediction)

        X = np.arange(0,9).reshape(1,3,3)

        print(X)
        print(np.reshape(X, (9)))
        print(model.predict(X))
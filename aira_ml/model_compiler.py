import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class ModelCompiler:

    @classmethod
    def compile_tf_model(cls, path_to_model):
        """Compiles a TensorFlow model to the current supported Aira objects.
        Currently supports Sequential models saved in the SavedModel format.
        """

        model = tf.keras.models.load_model(path_to_model)
        model.summary()

        for i, layer in enumerate(model.layers):

            if 'dense' in layer.name:
                cls.extract_dense(layer, i)
            elif 'flatten' in layer.name:
                cls.extract_flatten(layer, i)

    @classmethod
    def extract_dense(cls, layer, index):
        """This method compiles the data in a dense layer to a Dense
        Aira object.
        """

    def extract_flatten(cls, layer, index):
        """This method extracts relevant data from a Flatten layer.
        This method will be utilised when integrating aira-ml into 
        a larger framework, as all data is automatically flattened
        when passing through a serial interface to the FPGA.
        """
        pass

ModelCompiler.compile_tf_model("models/dense_mnist/model")

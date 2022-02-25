import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class ModelCompiler:

    @classmethod
    def compile_tf_model(cls, path_to_model):
        """Compiles a TensorFlow model to the current supported Aira objects
        """

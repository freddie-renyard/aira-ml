import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from aira_ml.tools.matrix_tools import MatrixTools
from aira_ml.aira_objects import DenseAira

class ModelCompiler:

    @classmethod
    def compile_tf_model(cls, path_to_model):
        """Compiles a TensorFlow model to the current supported Aira objects.
        Currently supports Sequential models saved in the SavedModel format.
        """

        model = tf.keras.models.load_model(path_to_model)
        model.summary()

        aira_sequential = []
        for i, layer in enumerate(model.layers):

            if 'dense' in layer.name:
                aira_sequential.append(cls.extract_dense(layer, i))
            elif 'flatten' in layer.name:
                cls.extract_flatten(layer, i)
        
        verilog_header = open("aira_ml/sv_source/header_source/aira_params.vh").read()

        # Compile the input data width
        first_obj = aira_sequential[0]
        if first_obj.input_is_floating:
            in_width = 1 + first_obj.n_input_exponent + first_obj.n_input_mantissa
        else:
            in_width = first_obj.n_input_mantissa

        # Compile the output data width
        last_obj = aira_sequential[-1]
        if last_obj.input_is_floating:
            out_width = 1 + last_obj.n_output_exponent + last_obj.n_output_mantissa
        else:
            out_width = last_obj.n_output_mantissa

        # Insert the data widths into the header file.
        verilog_header = verilog_header.replace("<n_mod_in>", str(in_width))
        verilog_header = verilog_header.replace("<n_mod_out>", str(out_width))

        # Loop over all the objects in the model.
        for aira_obj in aira_sequential:

            # Add the object's parameter declarations into 
            # the Verilog header.
            verilog_header += aira_obj.compile_verilog_header()

        with open("aira_ml/cache/aira_params.vh", 'w') as output_file:
            output_file.write(verilog_header)

    @classmethod
    def extract_dense(cls, layer, index):
        """This method compiles the data in a dense layer to a Dense
        Aira object.
        """

        weights, biases = layer.get_weights()

        weights = MatrixTools.sparsify_matrix_simple(weights, density=0.5)
        #MatrixTools.plot_histogram(weights)
        
        # Create the Aira Dense object, which will compile the data to
        # the representations used in the FPGA.
        dense_obj = DenseAira(
            index           = index,
            weights         = weights,
            biases          = biases, 
            act_name        = layer.activation.__qualname__,
            n_data_mantissa = 3,
            n_data_exponent = 3,
            n_input_mantissa= 3,
            n_input_exponent= 3,
            n_weight_mantissa= 3,
            n_weight_exponent= 3,
            n_output_mantissa= 3,
            n_output_exponent= 4,
            n_overflow       = 1,
            mult_extra       = 1
        )

        return dense_obj

    @classmethod
    def extract_flatten(cls, layer, index):
        """This method extracts relevant data from a Flatten layer.
        This method will be utilised when integrating aira-ml into 
        a larger framework, as all data is automatically flattened
        when passing through a serial interface to the FPGA.
        """
        pass

ModelCompiler.compile_tf_model("models/dense_mnist/model")

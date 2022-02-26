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
        
        cls.compile_full_header(aira_sequential)

        cls.compile_system_verilog(aira_sequential)

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

    @classmethod
    def compile_full_header(cls, aira_sequential):
        """Compiles the Verilog header file for the model. The Aira
        objects are passed in in a list.
        """
        # Open the start points of the Verilog header file.
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
    def compile_system_verilog(cls, aira_sequential):
        """ Compiles the full SystemVerilog source file.
        The Aira objects are passed in as a list.
        """

        aira_ml_top = open("aira_ml/sv_source/aira_ml_top.sv").read()

        # Add wire declarations to the Verilog
        for aira_obj in aira_sequential:
            aira_ml_top += aira_obj.compile_verilog_wires()

        aira_ml_top += cls.compile_connections(aira_sequential)

        # Add module declarations to the Verilog
        for aira_obj in aira_sequential:
            aira_ml_top += aira_obj.compile_verilog_module()

        # Terminate the aira_ml_top module
        aira_ml_top += "\nendmodule"

        with open("aira_ml/cache/aira_ml_top.sv", "w") as output_file:
            output_file.write(aira_ml_top)

    @classmethod
    def compile_connections(cls, aira_sequential):
        """Compiles the model's connection's into a list of assignments.
        Returns the Verilog to be added to the aira_ml top file as a string.
        """

        connections_str = ""
        
        # Add input connection to the Verilog
        obj_index = str(aira_sequential[0].index)
        output_str = open("aira_ml/sv_source/input_connection.sv").read()
        connections_str += output_str.replace("<i_post>", obj_index)

        # Add intermediate connections to the Verilog
        obj_num = len(aira_sequential)

        for i in range(1, obj_num-1):
            pre_index = str(aira_sequential[i-1].index)
            post_index = str(aira_sequential[i].index)

            output_str = open("aira_ml/sv_source/port_connection.sv").read()
            output_str = output_str.replace("<i_pre>", pre_index)
            connections_str += output_str.replace("<i_post>", post_index)

        # Add output connection to the Verilog
        obj_index = str(aira_sequential[-1].index)
        output_str = open("aira_ml/sv_source/output_connection.sv").read()
        connections_str += output_str.replace("<i_pre>", obj_index)

        return connections_str
    
ModelCompiler.compile_tf_model("models/dense_mnist/model")

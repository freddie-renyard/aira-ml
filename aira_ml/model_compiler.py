import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from aira_ml.tools.matrix_tools import MatrixTools
from aira_ml.aira_objects import DenseAira
import json
import os
from subprocess import check_call
from aira_ml.tools.aira_exceptions import AiraException

class ModelCompiler:

    @classmethod
    def compile_tf_model(cls, path_to_model):
        """Compiles a TensorFlow model to the current supported Aira objects.
        Currently supports Sequential models saved in the SavedModel format.
        """

        model = tf.keras.models.load_model(path_to_model)
        model.summary()

        aira_sequential = []

        # Get the compiler parameters
        with open("aira_ml/config/compiler_config.json") as file:
            params = json.load(file)

        prev_man = params["starting_mantissa"]
        prev_exp = params["starting_exponent"]

        # Count the number of Dense layers, so that the last
        # Dense layer index can be determined in the next stage.
        layer_count = 0
        for layer in model.layers:
            if 'dense' in layer.name:
                layer_count += 1

        index = 0
        for layer in model.layers:
            if 'dense' in layer.name:
                
                if index == (layer_count - 1):
                    multithread = False
                else:
                    multithread = True

                dense_obj, prev_man, prev_exp = cls.extract_dense(layer, index, prev_man, prev_exp, multithreading=multithread)
                
                aira_sequential.append(dense_obj)

                index += 1
            elif 'flatten' in layer.name:
                cls.extract_flatten(layer)

        # Extract the shape of the output tensor from the last layer.
        shape = np.array(layer.output_shape)
        output_shape = list(shape[shape != np.array(None)])
        
        cls.compile_full_header(aira_sequential)

        cls.compile_system_verilog(aira_sequential)

        cls.compile_serial_params(
            n_input     = aira_sequential[0].n_input,
            n_output    = aira_sequential[-1].n_output,
            input_num   = aira_sequential[0].pre_neuron_num,
            output_num  = aira_sequential[-1].post_neuron_num,
            in_format   = cls.determine_format(aira_sequential[0]),
            out_format  = cls.determine_format(aira_sequential[-1], get_input=False),
            n_in_man    = aira_sequential[0].n_input_mantissa,
            n_in_exp    = aira_sequential[0].n_input_exponent,
            n_out_man   = aira_sequential[-1].n_output_mantissa,
            n_out_exp   = aira_sequential[-1].n_output_exponent,
            output_shape = output_shape
        )

        cls.call_synthesis_server()

    @classmethod
    def extract_dense(cls, layer, index, n_in_mantissa, n_in_exponent, multithreading=False):
        """This method compiles the data in a dense layer to a Dense
        Aira object.
        """

        weights, biases = layer.get_weights()

        weights = MatrixTools.threshold_matrix(weights, threshold=0.1, verbose=True)
        #MatrixTools.plot_histogram(weights)

        # Get the compiler parameters
        with open("aira_ml/config/compiler_config.json") as file:
            params = json.load(file)

        # Determine the output depths
        out_mantissa = n_in_mantissa + params["mantissa_growth"]
        out_exponent = n_in_exponent + params["exponent_growth"]

        if multithreading:
            threads = params["threads"]
        else:
            threads = 1

        
        # Create the Aira Dense object, which will compile the data to
        # the representations used in the FPGA.
        dense_obj = DenseAira(
            index           = index,
            weights         = weights,
            biases          = biases, 
            act_name        = layer.activation.__qualname__,
            n_input_mantissa= n_in_mantissa,
            n_input_exponent= n_in_exponent,
            n_weight_mantissa= params["weight_mantissa"],
            n_weight_exponent= params["weight_exponent"],
            n_output_mantissa= out_mantissa,
            n_output_exponent= out_exponent,
            n_overflow       = params["n_overflow"],
            mult_extra       = params["mult_extra"],
            threads          = threads
        )

        return dense_obj, out_mantissa, out_exponent

    @classmethod
    def extract_flatten(cls, layer, index=0):
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

        for i in range(1, obj_num):
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

    @staticmethod
    def compile_serial_params(n_input, n_output, 
        input_num, output_num, 
        in_format, out_format,
        n_in_man, n_in_exp,
        n_out_man, n_out_exp,
        output_shape):
        """Compiles the parameters needed to run the serial interface.
        Nothing is returned; a JSON file is saved to the cache.
        """

        json_dict = {}

        # Save the bit depths of the input and output.
        json_dict["input_bit_depth"] = n_input
        json_dict["output_bit_depth"] = n_output

        # Save the number of entries to send and receive from the interface.
        json_dict["input_number"] = input_num
        json_dict["output_number"] = output_num

        if in_format == 'float' or in_format == 'int':
            json_dict["input_format"] = in_format
        else:
            raise AiraException("Input format is not recognised as float or int.")

        if out_format == 'float' or out_format == 'int':
            json_dict["output_format"] = out_format
        else:
            raise AiraException("Output format is not recognised as float or int.")

        json_dict["n_input_mantissa"] = n_in_man
        json_dict["n_input_exponent"] = n_in_exp

        json_dict["n_output_mantissa"] = n_out_man
        json_dict["n_output_exponent"] = n_out_exp

        json_dict["output_tensor_shape"] = output_shape

        with open("aira_ml/cache/serial_params.json", "w") as file:
            json.dump(json_dict, file)

    @staticmethod
    def determine_format(obj, get_input=True):
        """Determines the input or output format of an Aira object.
        Used for compiling the parameters needed for the serial link.
        """
        if get_input:
            exp_num = obj.n_input_exponent
        else:
            exp_num = obj.n_output_exponent
        
        if exp_num == 0:
            return 'int'
        else:
            return 'float'

    @staticmethod
    def call_synthesis_server():
        """Transfers the contents of the cache to the synthesis server.
        """

        with open("aira_ml/config/server_config.json") as file:
            server_config = json.load(file)

        cwd = os.getcwd()

        server_path = server_config["project_dir"]
        server_addr = server_config["ssh_addr"]
        vivado_loc = server_config["vivado_loc"]
        project_path = server_config["project_loc"]
        script_path = cwd + "/aira_ml/file_transfer.sh {} {} {} {}"

        check_call(script_path.format(server_path, server_addr, vivado_loc, project_path), shell=True)
    
ModelCompiler.compile_tf_model("models/dense_mnist/model")
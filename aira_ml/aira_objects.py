from tabnanny import verbose
import numpy as np
from aira_ml.tools.aira_exceptions import AiraException
from aira_ml.tools.binary_tools import BinCompiler
from aira_ml.tools.file_tools import Filetools
from math import ceil, log2

class DenseAira:

    def __init__(self, index, weights, biases, act_name, 
        n_input_mantissa, n_input_exponent,
        n_weight_mantissa, n_weight_exponent,
        n_output_mantissa, n_output_exponent,
        n_overflow, mult_extra,
        n_data_mantissa, n_data_exponent):
        
        self.index = index

        # Ensure that the activation function used in the layer has
        # hardware support.
        self.act_name = None
        if act_name == 'relu':
            self.act_name = act_name
        else:
            raise AiraException("Unsupported function found in a Dense layer: {}".format(act_name))

        # Infer the number of neurons from the dimensionality of the weight matrix
        weight_dims = np.shape(weights)
        self.pre_neuron_num = weight_dims[0]
        self.post_neuron_num = weight_dims[1]

        self.n_data_mantissa = n_data_mantissa
        self.n_data_exponent = n_data_exponent

        self.n_input_mantissa = n_input_mantissa
        self.n_input_exponent = n_input_exponent

        self.n_weight_mantissa = n_weight_mantissa
        self.n_weight_exponent = n_weight_exponent

        self.n_output_mantissa = n_output_mantissa
        self.n_output_exponent = n_output_exponent

        # Multiplier optimisation parameters - control of internal datapaths
        self.n_overflow = n_overflow
        self.mult_extra = mult_extra

        # Check that the weights are stored in a valid way.
        if np.shape(weight_dims)[0] != 2:
            raise AiraException("The weights for Dense layer {} are not stored in a 2D tensor.".format(index))

        # Compile the weights and biases.
        self.comp_weights = self.compile_mem(weights, biases, compile_delta=True)

        Filetools.save_to_file(
            "dense_weights_{}".format(index), 
            self.comp_weights, 
            verbose=False
        )

        self.compile_verilog_header()

    def compile_mem(self, weights, biases, compile_delta=False):
        """Compile binary strings to be saved/transferred to the FPGA.
        The addresses are delta encoded:
        """

        weights = np.transpose(weights)

        # Determine the biggest address change across the whole matrix.
        max_delta = 0

        for row in weights:
            non_zero_indices = np.squeeze(np.nonzero(row))
            row_deltas = np.diff(non_zero_indices)
            current_max = np.amax(row_deltas)
            max_delta = current_max if (current_max > max_delta) else max_delta

            # Check the size of the initial row index.
            current_max = non_zero_indices[0]
            max_delta = current_max if (current_max > max_delta) else max_delta
        
        # Add one to the address number to allow for the 'all ones' row break code in hardware.
        self.n_pre_addr = ceil(log2(max_delta + 1))
        self.n_memory = self.n_pre_addr + 1 + self.n_data_mantissa + self.n_data_exponent
        
        # Compile the weights.
        comp_weights = []

        # Compile a 'dummy' weight for the start of the memory file to signal the bias load
        full_entry = "1" * self.n_memory
        comp_weights.append(full_entry)

        for row, bias in zip(weights, biases):

            non_zero_indices = np.squeeze(np.nonzero(row))
            row_deltas = np.squeeze(np.diff(non_zero_indices))

            # Compile bias value with the starting address.
            full_entry = self.compile_row_data(bias, np.squeeze(non_zero_indices)[0])
            comp_weights.append(full_entry)
            
            if len(full_entry) != self.n_memory:
                raise AiraException("Compiler Error: Binary strings are unequal.")
            
            shifted_deltas = list(row_deltas[1:])
            shifted_deltas.append(0)

            for delta, data_i in zip(shifted_deltas, non_zero_indices):

                data_weight = row[data_i]

                # Signals the end of the row. Compile the row break signal (all ones)
                if delta == 0:
                    comp_weight = BinCompiler.compile_to_float(
                        data_weight,
                        self.n_data_mantissa,
                        self.n_data_exponent
                    )

                    break_code = "1" * self.n_pre_addr
                    full_entry = comp_weight + break_code
                    comp_weights.append(full_entry)
                else:
                    # Compile weight data and the change in address for next weight.
                    full_entry = self.compile_row_data(data_weight, delta)
                    comp_weights.append(full_entry)

        return comp_weights

    def compile_row_data(self, data, address):

        comp_weight = BinCompiler.compile_to_float(
            data, 
            n_mantissa = self.n_data_mantissa,
            n_exponent = self.n_data_exponent
        )

        comp_addr = BinCompiler.compile_to_uint(
            address,
            n_output = self.n_pre_addr,
            n_radix = 0
        )

        return comp_weight + comp_addr

    def compile_verilog_header(self):
        """Compile the parameters for the model into the Verilog header.
        """

        output_str = open("aira_ml/sv_source/header_source/dense_header.sv").read()

        # Replace the markup with the parameters
        output_str = output_str.replace("<pre_neurons>", str(self.pre_neuron_num))
        output_str = output_str.replace("<post_neurons>", str(self.post_neuron_num))

        output_str = output_str.replace("<n_man_input>", str(self.n_input_mantissa))
        output_str = output_str.replace("<n_exp_input>", str(self.n_input_exponent))

        output_str = output_str.replace("<n_man_weight>", str(self.n_weight_mantissa))
        output_str = output_str.replace("<n_exp_weight>", str(self.n_weight_exponent))

        output_str = output_str.replace("<n_man_out>", str(self.n_output_mantissa))
        output_str = output_str.replace("<n_exp_out>", str(self.n_output_exponent))

        output_str = output_str.replace("<n_overflow>", str(self.n_overflow))
        output_str = output_str.replace("<mult_extra>", str(self.mult_extra))

        mem_depth = len(self.comp_weights)
        output_str = output_str.replace("<mem_depth>", str(mem_depth))
        output_str = output_str.replace("<n_delta>", str(self.n_pre_addr))

        if self.act_name == 'relu':
            act_code = "1"
        else:
            act_code = "0"
        
        output_str = output_str.replace("<act_code>", act_code)

        output_str = output_str.replace("<index>", str(self.index))

        print(output_str)
        exit()



import numpy as np
from aira_ml.tools.aira_exceptions import AiraException
from aira_ml.tools.binary_tools import BinCompiler
from aira_ml.tools.file_tools import Filetools
from math import ceil, log2

class AiraLayer:

    def __init__(self, index, act_name, 
        n_input_mantissa, n_input_exponent,
        n_weight_mantissa, n_weight_exponent,
        n_output_mantissa, n_output_exponent,
        n_overflow, mult_extra
        ):
        # The parent class for other AiraML layers.

        self.index = index
        self.act_name = self.check_act_fn_support(act_name)

        # Determine datapath parameters.
        self.weight_params = {
            'n_man': n_weight_mantissa,
            'n_exp': n_weight_exponent,
            'n_data': 1 + n_weight_exponent + n_weight_mantissa
        }

        self.input_params = {
            'n_man': n_input_mantissa,
            'n_exp': n_input_exponent,
            'n_data': 1 + n_input_exponent + n_input_mantissa
        }

        self.output_params = {
            'n_man': n_output_mantissa,
            'n_exp': n_output_exponent,
            'n_data': 1 + n_output_exponent + n_output_mantissa
        }

        self.alu_params = {
            'mult_extra': mult_extra,
            'n_overflow': n_overflow
        }

    def check_act_fn_support(self, act_name):
        # Checks the support for activation functions passed to the layer.

        if act_name == 'relu':
            return act_name
        else:
            raise AiraException("Unsupported function found in a Dense layer: {}".format(act_name))

    def compile_common_header(self):
        # Compiles parameters that are common to all Aira layers.
        
        output_str = open("aira_ml/sv_source/header_source/common_header.sv").read()
        output_str = output_str.replace("<i>", str(self.index))

        # Replace the markup with the parameters
        output_str = output_str.replace("<n_man_input>", str(self.input_params['n_man']))
        output_str = output_str.replace("<n_exp_input>", str(self.input_params['n_exp']))
        output_str = output_str.replace("<n_input>", str(self.input_params['n_data']))

        output_str = output_str.replace("<n_man_weight>", str(self.weight_params['n_man']))
        output_str = output_str.replace("<n_exp_weight>", str(self.weight_params['n_exp']))

        output_str = output_str.replace("<n_man_out>", str(self.output_params['n_man']))
        output_str = output_str.replace("<n_exp_out>", str(self.output_params['n_exp']))
        output_str = output_str.replace("<n_output>", str(self.output_params['n_data']))

        output_str = output_str.replace("<n_overflow>", str(self.alu_params['n_overflow']))
        output_str = output_str.replace("<mult_extra>", str(self.alu_params['mult_extra']))

        if self.act_name == 'relu':
            act_code = "1"
        else:
            act_code = "0"
        
        output_str = output_str.replace("<act_code>", act_code)

        return output_str
    
    def get_comma_delim_lst(self, target_lst):
        return ','.join(target_lst)
        
class DenseAira(AiraLayer):

    def __init__(self, index, weights, biases, act_name, 
        n_input_mantissa, n_input_exponent,
        n_weight_mantissa, n_weight_exponent,
        n_output_mantissa, n_output_exponent,
        n_overflow, mult_extra,
        threads):

        # Initialise the parent layer params
        super().__init__(index, act_name, 
            n_input_mantissa, n_input_exponent,
            n_weight_mantissa, n_weight_exponent,
            n_output_mantissa, n_output_exponent,
            n_overflow, mult_extra
        )

        # NB The non-floating point support has been removed.

        # Infer the number of neurons from the dimensionality of the weight matrix
        weight_dims = np.shape(weights)
        self.pre_neuron_num = weight_dims[0]
        self.post_neuron_num = weight_dims[1]

        self.threads = self.verify_thread_validity(threads, self.post_neuron_num)

        # Check that the weights are stored in a valid way.
        if np.shape(weight_dims)[0] != 2:
            raise AiraException("The weights for Dense layer {} are not stored in a 2D tensor.".format(index))

        # Compile the weights and biases.
        weights = np.transpose(weights)
        self.mem_depths = []
        self.n_pre_addr = self.determine_mem_depth(weights)

        if self.threads == 1:
            compiled_weights = self.compile_mem(weights, biases)

            Filetools.save_to_file(
                "dense_weights_{}_thread_{}".format(index, 0), 
                compiled_weights, 
                verbose=False
            )

            self.mem_depths.append(len(compiled_weights))
        else:
            # TODO Put this into a function.

            neurons_per_thread = self.post_neuron_num // self.threads

            # Create an empty 3D numpy array with shape: 
            # (number of threads, neurons_per_thread, pre neuron number)
            interlaced_shape = (self.threads, neurons_per_thread, self.pre_neuron_num)
            weights_interlaced = np.zeros(interlaced_shape)

            # Create an empty 2D numpy array for the interlaced biases.
            biases_shape = (threads, neurons_per_thread)
            biases_interlaced = np.zeros(biases_shape)

            # Interlace the weight matrix into seperate matrices.
            for i, data in enumerate(zip(weights, biases)):
                weights_interlaced[i % self.threads][i // self.threads] = data[0]
                biases_interlaced[i % self.threads][i // self.threads] = data[1]

            # Compile the data. TODO Combine with above loop.
            for thread_i, data in enumerate(zip(weights_interlaced, biases_interlaced)):
                compiled_weights = self.compile_mem(data[0], data[1])
                self.mem_depths.append(len(compiled_weights))
                Filetools.save_to_file(
                    "dense_weights_{}_thread_{}".format(index, thread_i), 
                    self.compile_mem(data[0], data[1]), 
                    verbose=False
                )

    def determine_mem_depth(self, weights):
        """Determine the biggest address change across a matrix.
        """

        max_delta = 0
        for row in weights:
            non_zero_indices = np.squeeze(np.nonzero(row))
            row_deltas = np.diff(non_zero_indices)
            current_max = np.amax(row_deltas)
            max_delta = current_max if (current_max > max_delta) else max_delta

            # Check the size of the initial row index.
            current_max = non_zero_indices[0]
            max_delta = current_max if (current_max > max_delta) else max_delta
        
        # Add two to the address number to allow for the 'all ones' row break code in hardware.
        return ceil(log2(max_delta + 2))

    def compile_mem(self, weights, biases):
        """Compile binary strings to be saved/transferred to the FPGA.
        The addresses are delta encoded:
        """
        
        n_memory = self.n_pre_addr + 1 + self.weight_params['n_exp'] + self.weight_params['n_man']
        
        # Compile the weights.
        comp_weights = []

        # Compile a 'dummy' weight for the start of the memory file to signal the bias load
        full_entry = "1" * n_memory
        comp_weights.append(full_entry)

        for row, bias in zip(weights, biases):
                        
            non_zero_indices = np.squeeze(np.nonzero(row))
            row_deltas = np.squeeze(np.diff(non_zero_indices))

            # Compile bias value with the starting address.
            full_entry = self.compile_row_data(bias, np.squeeze(non_zero_indices)[0])
            comp_weights.append(full_entry)
            
            if len(full_entry) != n_memory:
                raise AiraException("Compiler Error: Binary strings are unequal.")
            
            shifted_deltas = list(row_deltas)
            shifted_deltas.append(0)

            for delta, data_i in zip(shifted_deltas, non_zero_indices):

                data_weight = row[data_i]
                
                # Signals the end of the row. Compile the row break signal (all ones)
                if delta == 0:
                    comp_weight = BinCompiler.compile_to_float(
                        data_weight,
                        self.weight_params['n_man'],
                        self.weight_params['n_exp']
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
            n_mantissa = self.weight_params['n_man'],
            n_exponent = self.weight_params['n_exp']
        )

        comp_addr = BinCompiler.compile_to_uint(
            address,
            n_output = self.n_pre_addr,
            n_radix = 0
        )

        return comp_weight + comp_addr

    def verify_thread_validity(self, threads, neurons):
        """Determine if the number of threads is able to be
        realised in hardware.
        """

        threads_invalid = True
        while threads_invalid:
            if neurons % threads == 0:
                threads_invalid = False
            else:
                print("{} threads cannot run {} neurons. Decreasing thread number...".format(threads, neurons))
                threads -= 1
        
        return threads

    def compile_layer_header(self):
        """Compile the parameters for the model into the Verilog header.
        """

        output_str = open("aira_ml/sv_source/header_source/dense_header.sv").read()
        output_str = output_str.replace("<i>", str(self.index))
        
        # Replace the markup with the parameters
        output_str = output_str.replace("<pre_neurons>", str(self.pre_neuron_num))
        output_str = output_str.replace("<post_neurons>", str(self.post_neuron_num))

        output_str = output_str.replace("<n_delta>", str(self.n_pre_addr))
        output_str = output_str.replace("<threads>", str(self.threads))

        return output_str

    def compile_verilog_wires(self):
        """Insert the marked-up wire declarations into the Verilog source code.
        Returns the SystemVerilog code as a string.
        """

        output_str = open("aira_ml/sv_source/dense_wires.sv").read()
        
        return output_str.replace("<i>", str(self.index))

    def compile_verilog_module(self):
        """Insert the marked-up module declaration into the Verilog source code.
        Returns the SystemVerilog code as a string.
        """

        output_str = open("aira_ml/sv_source/dense_module.sv").read()

        self.mem_depths = self.mem_depths[::-1]
        depth_list = str(self.mem_depths[0])
        for depth_val in self.mem_depths[1:-1]:
            depth_list += ", {}".format(depth_val)

        if len(self.mem_depths) != 1:
            depth_list += ", {}".format(self.mem_depths[-1]) 

        output_str = output_str.replace("<thread-list>", depth_list)
        return output_str.replace("<i>", str(self.index))

class Conv2DMaxPoolAira(AiraLayer):

    def __init__(self, index, filters, biases, act_name, 
        n_input_mantissa, n_input_exponent,
        n_weight_mantissa, n_weight_exponent,
        n_output_mantissa, n_output_exponent,
        n_overflow, mult_extra, 
        input_shape, max_pool,
        filter_threads, rowcol_threads, channel_threads
        ):

        # Initialise the parent layer params
        super().__init__(index, act_name, 
            n_input_mantissa, n_input_exponent,
            n_weight_mantissa, n_weight_exponent,
            n_output_mantissa, n_output_exponent,
            n_overflow, mult_extra
        )

        # Determine tensor parameters for the convolution
        conv_tensor_shape       = np.shape(filters)
        self.filter_num         = conv_tensor_shape[3]
        self.prelayer_channels  = conv_tensor_shape[2]
        self.kernel_dim         = conv_tensor_shape[0]

        # Determine input image shape
        self.input_shape = input_shape
        self.max_pool = max_pool

        # Determine the parallelisation parameters.
        self.filter_threads = filter_threads # The number of threads used to compute the filter
        self.rowcol_threads = rowcol_threads # The number of threads used within each convolution on an image
        self.channel_threads = channel_threads
        
        if self.channel_threads is not None:
            if self.channel_threads != self.prelayer_channels:
                raise AiraException("The number of channel threads must be the same as the number of channels in the input tensor ({}).".format(prelayer_channels))

        # Compile filters.
        weight_dat = self.allocate_filters(filters)
        self.compile_weights(weight_dat, concat_channels=True)

        # Compile biases. 
        self.allocate_and_compile_biases(biases)

        # Compile entry and exit pointers.
        self.compile_pointers()

    def allocate_filters(self, filters):
        
        # Flatten kernels into 1D representations.
        filt_shape = np.shape(filters)
        kernel_num = filt_shape[-1]
        new_shape = (filt_shape[0] * filt_shape[1], *filt_shape[2:])
        filters = np.array(filters).reshape(*new_shape)
        
        # Allocate kernels to filter threads.
        kerns_per_thread = filt_shape[-1] / self.filter_threads
        if not kerns_per_thread.is_integer():
            raise AiraException("Number of kernels ({} kernels) cannot be evenly divided amongst specified threads ({} threads).".format(kernel_num, self.filter_threads))
        kerns_per_thread = int(kerns_per_thread)
        
        weight_struct = [[[] for _ in range(self.channel_threads)] for _ in range(self.filter_threads)]
        for i_kern in range(kernel_num):
            for i_chan in range(self.channel_threads):
                chan_filt = filters[:, i_chan, i_kern]
                weight_struct[i_kern // kerns_per_thread][i_chan] += list(chan_filt)
            
        return weight_struct

    def compile_weights(self, weight_dat, concat_channels=True):
        
        for i_kern, filter_weights in enumerate(weight_dat):
            
            float_dat = []
            for i_chan, channel_weights in enumerate(filter_weights):
                new_float_dat = [BinCompiler.compile_to_float(
                    x, 
                    n_mantissa = self.weight_params['n_man'],
                    n_exponent = self.weight_params['n_exp']
                ) for x in channel_weights]

                if concat_channels:
                    if len(float_dat) == 0:
                        float_dat = new_float_dat
                    else:
                        float_dat = [x[0]+x[1] for x in zip(new_float_dat, float_dat)]
                else:
                    Filetools.save_to_file(
                        "conv2dmaxpool_{}_weights_filtthread_{}_chanthread_{}".format(self.index, i_kern, i_chan), 
                        new_float_dat, 
                        verbose=False
                    )
  
            if concat_channels:
                Filetools.save_to_file(
                    "conv2dmaxpool_{}_weights_filtthread_{}_concat".format(self.index, i_kern, i_chan), 
                    float_dat, 
                    verbose=False
                )

    def allocate_and_compile_biases(self, biases, concat=True):

        kerns_per_thread = np.shape(biases)[0] / self.filter_threads
        if kerns_per_thread.is_integer() == False:
            raise AiraException("The number of biases is not evenly divisible by the kernel number.")
        kerns_per_thread = int(kerns_per_thread)        
        
        float_dat = []
        for i in range(self.filter_threads):

            bin_float_dat = [BinCompiler.compile_to_float(
                x, 
                n_mantissa = self.weight_params['n_man'],
                n_exponent = self.weight_params['n_exp']
            ) for x in biases[i * kerns_per_thread: (i+1) * kerns_per_thread]]

            if concat:
                if len(float_dat) == 0:
                    float_dat = bin_float_dat
                else:
                    float_dat = [x[0]+x[1] for x in zip(bin_float_dat, float_dat)]
            else:
                Filetools.save_to_file(
                    "conv2dmaxpool_{}_biases_filtthread_{}".format(self.index, i), 
                    bin_float_dat, 
                    verbose=False
                )
        
        if concat:
            Filetools.save_to_file(
                "conv2dmaxpool_{}_biases".format(self.index), 
                float_dat, 
                verbose=False
            )

    def compile_pointers(self):
        # Compiles the entry and exit pointers for the rowcol threads.

        self.entry_ptrs = []
        self.exit_ptrs = []
        pass

    def compile_layer_header(self):
        """Compile the parameters for the model into the Verilog header.
        """

        output_str = open("aira_ml/sv_source/header_source/conv2d_header.sv").read()
        output_str = output_str.replace("<i>", str(self.index))
        
        # Replace the markup with the parameters
        output_str = output_str.replace("<n_input_ports>", str(self.prelayer_channels))
        output_str = output_str.replace("<n_chan>", str(self.prelayer_channels))
        output_str = output_str.replace("<n_thread_chan>", str(self.prelayer_channels))
        output_str = output_str.replace("<n_filter>", str(self.filter_num))
        output_str = output_str.replace("<filter_dim>", str(self.kernel_dim))
        output_str = output_str.replace("<max_pool>", str(self.max_pool))

        output_str = output_str.replace("<n_col>", str(self.input_shape[1]))
        output_str = output_str.replace("<n_row>", str(self.input_shape[0]))

        output_str = output_str.replace("<n_thread_filter>", str(self.filter_threads))
        output_str = output_str.replace("<n_thread_filter>", str(self.rowcol_threads))

        output_str = output_str.replace("<entry_ptrs>", ','.join(self.entry_ptrs))
        output_str = output_str.replace("<exit_ptrs>", ','.join(self.exit_ptrs))
        
        return output_str
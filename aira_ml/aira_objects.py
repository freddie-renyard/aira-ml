import numpy as np
from aira_ml.tools.aira_exceptions import AiraException
from aira_ml.tools.binary_tools import BinCompiler
from aira_ml.tools.file_tools import Filetools
from aira_ml.tools.lut_compiler import compile_sigmoid
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

        self.act_lookup = {
            'none': "0",
            'relu': "1",
            'sigmoid': "2"
        }

        self.lut_depth = 2 ** 8

        self.act_name = self.check_act_fn_support(act_name)

    def check_act_fn_support(self, act_name):
        # Checks the support for activation functions passed to the layer.

        if act_name == 'relu':
            return act_name
        if act_name == 'sigmoid':
            self.compile_sigmoid_lut()
            return act_name
        else:
            raise AiraException("Unsupported function found in a Dense layer: {}".format(act_name))

    def modify_threads(self, ops, threads):
        # Calculates the nearest number of threads that can compute the operation specified.
        # Used for thread allocation.

        
        if ops % threads == 0:
            return threads

        factors = set()
        for i in range(1, ops):
            if ops % i == 0:
                factors.add(i)
        
        factors = list(factors)
        factors.sort()
        
        if len(factors) == 0:
            factors = [1]
        
        difs = [abs(threads - fac) for fac in factors]
        return factors[difs.index(min(difs))]

    def compile_common_header(self):
        # Compiles parameters that are common to all Aira layers.
        
        output_str = open("aira_ml/sv_source/header_source/common_header.sv").read()
        output_str = output_str.replace("<i>", str(self.index))

        # Replace the markup with the parameters
        output_str = output_str.replace("<n_man_input>", str(self.input_params['n_man']))
        output_str = output_str.replace("<n_exp_input>", str(self.input_params['n_exp']))
        output_str = output_str.replace("<n_input>", str(self.input_params['n_data']))
        output_str = output_str.replace("<n_input_ports>", str(self.input_ports))
        output_str = output_str.replace("<input_len>", str(self.input_len))

        output_str = output_str.replace("<n_man_weight>", str(self.weight_params['n_man']))
        output_str = output_str.replace("<n_exp_weight>", str(self.weight_params['n_exp']))

        output_str = output_str.replace("<n_man_out>", str(self.output_params['n_man']))
        output_str = output_str.replace("<n_exp_out>", str(self.output_params['n_exp']))
        output_str = output_str.replace("<n_output>", str(self.output_params['n_data']))
        output_str = output_str.replace("<n_output_ports>", str(self.output_ports))
        output_str = output_str.replace("<output_len>", str(self.output_len))

        output_str = output_str.replace("<n_overflow>", str(self.alu_params['n_overflow']))
        output_str = output_str.replace("<mult_extra>", str(self.alu_params['mult_extra']))

        output_str = output_str.replace("<lut_depth>", str(self.lut_depth))
        output_str = output_str.replace("<act_code>", self.act_lookup[self.act_name])

        return output_str
    
    def compile_verilog_wires(self, file_name='layer_wires.sv'):
        """Insert the marked-up wire declarations into the Verilog source code.
        Returns the SystemVerilog code as a string.
        """
        output_str = open("aira_ml/sv_source/{}".format(file_name)).read()
        return output_str.replace("<i>", str(self.index))
    
    def compile_verilog_module(self):
        """Insert the marked-up module declaration into the Verilog source code.
        Returns the SystemVerilog code as a string.
        """
        output_str = open("aira_ml/sv_source/{}_module.sv".format(self.layer_name)).read()
        return output_str.replace("<i>", str(self.index))

    def compile_sigmoid_lut(self):
        """Compile a sigmoid lookup table for use in activation functions.
        """

        lut = compile_sigmoid(
            self.output_params['n_man'],
            self.output_params['n_exp'],
            lut_depth = self.lut_depth
        )

        Filetools.save_to_file(
            "sigmoid_lut_{}".format(self.index), 
            lut, 
            verbose=False
        )
    
    def report_finished(self):
        print("AIRA: Layer {} ({}) compiled successfully.".format(self.index, self.layer_name))

class DenseAira(AiraLayer):

    def __init__(self, index, weights, biases, act_name, 
        n_input_mantissa, n_input_exponent,
        n_weight_mantissa, n_weight_exponent,
        n_output_mantissa, n_output_exponent,
        n_overflow, mult_extra, input_ports,
        threads):

        # Initialise the parent layer params
        super().__init__(index, act_name, 
            n_input_mantissa, n_input_exponent,
            n_weight_mantissa, n_weight_exponent,
            n_output_mantissa, n_output_exponent,
            n_overflow, mult_extra
        )

        self.layer_name = 'dense'

        # Set the number of i/o ports
        self.input_ports = input_ports
        self.output_ports = 1

        # Infer the number of neurons from the dimensionality of the weight matrix
        weight_dims = np.shape(weights)
        self.input_len = weight_dims[0]
        self.output_len = weight_dims[1]

        # Ensure that the thread number is supported by the target.
        self.threads = self.modify_threads(self.output_len, threads)

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
            self.compile_thread_data(index, weights, biases)
        
        self.report_finished()

    def compile_thread_data(self, index, weights, biases):

        neurons_per_thread = self.output_len // self.threads

        # Create an empty 3D numpy array with shape: 
        # (number of threads, neurons_per_thread, pre neuron number)
        interlaced_shape = (self.threads, neurons_per_thread, self.input_len)
        weights_interlaced = np.zeros(interlaced_shape)

        # Create an empty 2D numpy array for the interlaced biases.
        biases_shape = (self.threads, neurons_per_thread)
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

    def compile_layer_header(self):
        """Compile the parameters for the model into the Verilog header.
        """

        output_str = open("aira_ml/sv_source/header_source/dense_header.sv").read()
        output_str = output_str.replace("<i>", str(self.index))
        
        # Replace the markup with the parameters
        output_str = output_str.replace("<pre_neurons>", str(self.input_len))
        output_str = output_str.replace("<post_neurons>", str(self.output_len))

        output_str = output_str.replace("<n_delta>", str(self.n_pre_addr))
        output_str = output_str.replace("<threads>", str(self.threads))

        self.mem_depths = self.mem_depths[::-1]
        depth_list = str(self.mem_depths[0])
        for depth_val in self.mem_depths[1:-1]:
            depth_list += ", {}".format(depth_val)

        if len(self.mem_depths) != 1:
            depth_list += ", {}".format(self.mem_depths[-1]) 

        output_str = output_str.replace("<thread-list>", depth_list)

        return output_str

class Conv2DMaxPoolAira(AiraLayer):

    def __init__(self, index, conv_layer, max_pool_layer,
        n_input_mantissa, n_input_exponent,
        n_weight_mantissa, n_weight_exponent,
        n_output_mantissa, n_output_exponent,
        n_overflow, mult_extra,
        filter_threads, rowcol_threads, channel_threads,
        override_z_addr = True
        ):

        # Initialise the parent layer params
        super().__init__(index, conv_layer.activation.__qualname__,
            n_input_mantissa, n_input_exponent,
            n_weight_mantissa, n_weight_exponent,
            n_output_mantissa, n_output_exponent,
            n_overflow, mult_extra
        )

        self.layer_name = 'conv2d'

        # Determine tensor parameters for the convolution
        conv_tensor_shape       = np.shape(conv_layer.weights[0])
        self.filter_num         = conv_tensor_shape[3]
        self.prelayer_channels  = conv_tensor_shape[2]
        self.kernel_dim         = conv_tensor_shape[0]

        # Determine input and output data entries
        self.input_len  = int(np.prod(conv_layer.input_shape[1:]))
        if max_pool_layer is not None:
            self.output_len = int(np.prod(max_pool_layer.output_shape[1:]))
        else:
            self.output_len = int(np.prod(conv_layer.output_shape[1:]))

        # Determine input image shape
        self.input_shape = conv_layer.input_shape[1:]
        self.max_pool = self.z_addr = (max_pool_layer is not None)

        padding_lookup = {
            'same': 1,
            'valid': 0
        }

        if conv_layer.padding not in padding_lookup.keys():
            raise AiraException("Convolution layers with padding of type {} are not currently supported.".format(conv_layer.padding))
        else:
            self.padding = padding_lookup[conv_layer.padding]
            
        self.z_addr = override_z_addr or self.z_addr
        
        # Check input layer strides
        if np.prod(conv_layer.strides) != 1:
            raise AiraException("Convolution layers with a stride > 1 are currently not supported.")
        
        if max_pool_layer is not None:
            if tuple(max_pool_layer.strides) != (2, 2):
                raise AiraException("Only 2x2 max pooling layers are currently supported.")

        # Set the number of i/o ports
        self.input_ports = self.prelayer_channels
        self.output_ports = self.filter_num

        # Determine the parallelisation parameters.
        self.filter_threads = filter_threads # The number of threads used to compute the filter
        self.rowcol_threads = rowcol_threads # The number of threads used within each convolution on an image
        
        if channel_threads is not None:
            if channel_threads != self.prelayer_channels:
                raise AiraException("The number of channel threads must be the same as the number of channels in the input tensor ({}).".format(prelayer_channels))

        self.channel_threads = self.prelayer_channels

        # Compile filters.
        weight_dat = self.allocate_filters(conv_layer.weights[0])
        self.compile_weights(weight_dat, concat_channels=True)

        # Compile biases. 
        self.allocate_and_compile_biases(np.array(conv_layer.bias))

        # Compile entry and exit pointers.
        if self.padding:
            padding = (int(self.kernel_dim / 2))
        else:
            padding = 0

        self.entry_ptrs, self.exit_ptrs = self.compile_pointers(z_addr=self.z_addr, padding=padding)

        # Compile thread output base addresses.
        if max_pool_layer is not None:
            self.out_base_addrs = self.compile_out_base_addrs(max_pool_layer, self.rowcol_threads)
        else:
            self.out_base_addrs = self.compile_out_base_addrs(conv_layer, self.rowcol_threads)

        # Compile thread address translation lookup table.
        self.compile_input_addr_lut()

        self.report_finished()
    
    def compile_input_addr_lut(self):
        
        n_row = self.input_shape[0]
        n_row_padded = n_row
        if self.padding:
            n_row_padded += 2 * int(self.kernel_dim / 2)

        n_img = np.prod(self.input_shape)

        lut     = np.zeros(n_img)
        lut[0]  = n_row_padded + int(self.kernel_dim / 2)

        row_ctr = 0
        for i in range(1, np.prod(n_img)):
            if i % n_row == 0:
                row_ctr = 0
                lut[i] = lut[i-1] + 2 * int(self.kernel_dim / 2) + 1
            else:
                row_ctr += 1
                lut[i] = lut[i-1] + 1

        # Compile LUT to unsigned binary
        
        int_dat = [BinCompiler.compile_to_uint(x, ceil(log2(self.input_len)), 0) for x in lut]

        Filetools.save_to_file(
            "conv2dmaxpool_{}_addr_lut".format(self.index), 
            int_dat, 
            verbose=False
        )

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

    def compile_pointers(self, z_addr=True, padding=0):
        # Compiles the entry pointers for the rowcol threads.
        
        shape_2d = (self.input_shape[0] + 2 * padding, self.input_shape[1] + 2 * padding)

        entry_points = int(np.prod(shape_2d))
        
        if z_addr:
            entry_mat = np.indices(shape_2d) % 2 == 0
            entry_mat = np.prod(entry_mat, axis=0)

            # Remove illegal entry points
            illegal_row = np.zeros(np.shape(entry_mat), dtype=int)
            for i, row in enumerate(illegal_row):
                if i < np.shape(illegal_row)[0] - self.kernel_dim:
                    illegal_row[i] += 1
            
            entry_mat = np.multiply(entry_mat, illegal_row) # Remove illegal row entry points
            entry_mat = np.multiply(entry_mat, np.transpose(illegal_row)) # Remove illegal column entry points

            # Compute a 2D map of the exit pointer locations.
            entry_coords = np.argwhere(entry_mat == 1)
            exit_coords  = [[x + self.kernel_dim, y + self.kernel_dim] for x, y in entry_coords]

            exit_mat = np.zeros(shape_2d, dtype=int)
            for coord in exit_coords:
                exit_mat[coord[0], coord[1]] = 1
        else:
            raise AiraException("Non-max pooling layers are not supported yet.")
            entry_mat = np.ones(shape_2d)
                
        lin_indices = np.reshape(entry_mat, np.prod(shape_2d), order="C") 
        entry_addrs = np.squeeze(np.where(lin_indices == 1))
        
        lin_indices = np.reshape(exit_mat, np.prod(shape_2d), order="C") 
        exit_addrs = np.squeeze(np.where(lin_indices == 1))

        try:
            entry_points = len(entry_addrs)
        except:
            entry_points = 1
            entry_addrs = [entry_addrs]
            exit_addrs = [exit_addrs]

        if self.rowcol_threads > entry_points:
            print(
                "AIRA: The number of rowcol threads requested ({}) is larger than the number of input pixels ({})."
                .format(self.rowcol_threads, entry_points)
            )
            self.rowcol_threads = entry_points
            print("AIRA: Set the rowcol thread number to the number of entry points ({})".format(entry_points))

        if entry_points % self.rowcol_threads:

            print(
                "AIRA: The input image which has {} entry points cannot be evenly divided between {} rowcol threads in layer {}."
                .format(entry_points, self.rowcol_threads, self.index)
            )

            self.rowcol_threads = self.modify_threads(entry_points, self.rowcol_threads)

            print("AIRA: Updated rowcol threads for layer {} to {} threads.".format(self.index, self.rowcol_threads))

        addr_incr = int(entry_points / self.rowcol_threads)

        entry_ptrs = []
        exit_ptrs  = []
        for i in range(0, self.rowcol_threads * addr_incr, addr_incr):
            entry_ptrs.append(entry_addrs[i])
            exit_ptrs.append(exit_addrs[i+addr_incr-1] + 1)
        
        return entry_ptrs, exit_ptrs
    
    def compile_out_base_addrs(self, layer, threads):

        sq_shape = layer.output_shape[1:3]
        n_sq = np.prod(sq_shape)
        offset = n_sq / threads

        return [int(x * offset) for x in range(threads)]

    def compile_layer_header(self):
        """Compile the parameters for the model into the Verilog header.
        """

        output_str = open("aira_ml/sv_source/header_source/conv2d_header.sv").read()
        output_str = output_str.replace("<i>", str(self.index))
        
        # Replace the markup with the parameters
        output_str = output_str.replace("<n_chan>", str(self.prelayer_channels))
        output_str = output_str.replace("<n_thread_chan>", str(self.prelayer_channels))
        output_str = output_str.replace("<n_filter>", str(self.filter_num))
        output_str = output_str.replace("<filter_dim>", str(self.kernel_dim))
        output_str = output_str.replace("<max_pool>", str(int(self.max_pool)))

        output_str = output_str.replace("<n_col>", str(self.input_shape[1]))
        output_str = output_str.replace("<n_row>", str(self.input_shape[0]))

        output_str = output_str.replace("<n_thread_filter>", str(self.filter_threads))
        output_str = output_str.replace("<n_thread_rowcol>", str(self.rowcol_threads))

        output_str = output_str.replace("<entry_ptrs>", ','.join([str(x) for x in self.entry_ptrs]))
        output_str = output_str.replace("<exit_ptrs>", ','.join([str(x) for x in self.exit_ptrs]))
    
        output_str = output_str.replace("<padding>", str(self.padding))
        output_str = output_str.replace("<out_base_addrs>", ','.join([str(x) for x in self.out_base_addrs]))

        return output_str

from aira_ml.tools.binary_tools import BinCompiler
from math import exp, ceil, log2

class LUTCompiler:

    @classmethod
    def compile_sigmoid(cls, n_mantissa_in, n_exp_in):
        """ Compile an lookup table for a sigmoid function with
        floating point inputs and unsigned fixed point outputs.

        Returns a sorted list of binary strings with addresses
        corresponding to the input data values.

        NB These values must be sign extended in hardware with a 0.
        """

        def sigmoid(x):
            return 1 / (1 + exp(-x))
        
        n_input = 1 + n_mantissa_in + n_exp_in
        lut_res = 2 ** n_input

        # Determine all the possible input binary numbers
        in_bin_str = [BinCompiler.compile_to_uint(x, n_input, 0) for x in range(0, lut_res)]

        # Determine what value this binary actually represents under 
        # floating point interpretation.
        in_float_vals = [BinCompiler.decode_custom_float(x, n_mantissa_in, n_exp_in) for x in in_bin_str]
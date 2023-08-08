from ast import expr
from aira_ml.tools.aira_exceptions import AiraException
from aira_ml.tools.binary_tools import BinCompiler
from math import exp, ceil, log2
from bitstring import Bits


def mid_float_convert(bin_str, n_man):
    """Decodes the custom floats used in the sigmoid lookup table.
    """

    sign = bin_str[0]
    exp = bin_str[1:5]
    man = "1" + bin_str[5:]

    man_val = int(man, 2) / (2 ** len(man))
    
    if sign == '0':
        sign_val = 1
    else:
        sign_val = -1

    exp_val = Bits(bin=exp).int - 4

    return sign_val * man_val * 2 ** exp_val

def compile_sigmoid(n_mantissa_in, n_exp_in, lut_depth, n_output=0, dtype='float'):
    """ Compiles an lookup table for a sigmoid function with
    floating point inputs and either unsigned fixed point outputs or 
    floating point outputs, with the same parameters as the input float.

    Returns a sorted list of binary strings with addresses
    corresponding to the input data values.

    NB These values must be sign extended in hardware with a 0.
    """

    def sigmoid(x):
        return 1 / (1 + exp(-x))
    
    lut_res = lut_depth
    bit_depth = ceil(log2(lut_depth))

    exp_limits = [4, -12] 
    exp_bits = ceil(log2(exp_limits[0] - exp_limits[1])) # Number of bits used to express the limited exponent range
    
    remaining_bits = bit_depth - exp_bits - 1 # See how many bits are left for mantissa lookup
    
    # Determine all the possible input binary numbers
    in_bin_str = [BinCompiler.compile_to_uint(x, bit_depth, 0) for x in range(0, lut_res)]

    # Determine what value this binary actually represents under 
    # floating point interpretation.
    in_float_vals = [mid_float_convert(x, remaining_bits) for x in in_bin_str]

    # Compute the sigmoid value for each value.
    out_sigmoid_vals = [sigmoid(x) for x in in_float_vals]

    if dtype == 'int':
        out_bin = [BinCompiler.compile_to_uint(x, n_output, n_output) for x in out_sigmoid_vals] 
    elif dtype == 'float':
        out_bin = [BinCompiler.compile_to_float(x, n_mantissa_in, n_exp_in) for x in out_sigmoid_vals] 
    else:
        raise AiraException("AIRA: This datatype ({}) is not supported by the sigmoid LUT compiler.".format(dtype))

    return out_bin

def compile_exponential(n_mantissa_in, n_exp_in, n_mantissa_out, n_exp_out):
    """ Compiles an lookup table for the exponential function with
    floating point inputs and floating point outputs.

    Returns a sorted list of binary strings with addresses
    corresponding to the input data values.
    """
    
    n_input = 1 + n_mantissa_in + n_exp_in
    lut_res = 2 ** n_input

    # Determine all the possible input binary numbers
    in_bin_str = [BinCompiler.compile_to_uint(x, n_input, 0) for x in range(0, lut_res)]

    # Determine what value this binary actually represents under 
    # floating point interpretation.
    in_float_vals = [BinCompiler.decode_custom_float(x, n_mantissa_in, n_exp_in) for x in in_bin_str]

    # Compute the exponential value for each value.
    out_exp_vals = [exp(x) for x in in_float_vals]

    out_bin = [BinCompiler.compile_to_float(x, n_mantissa_out, n_exp_out) for x in out_exp_vals] 

    return out_bin
from aira_ml.tools.binary_tools import BinCompiler
from math import exp, ceil, log2

class LUTCompiler:

    @classmethod
    def compile_sigmoid(cls, resolution):
        """ Compile an lookup table for a sigmoid function with
        floating point inputs and unsigned fixed point outputs.

        Returns a sorted list of binary strings with addresses
        corresponding to the input data values.

        NB These values must be sign extended in hardware with a 0.
        """

        def sigmoid(x):
            return 1 / (1 + exp(-x))
        
        # Truncate the input resolution to a power of two
        pow_two_res = ceil(log2(resolution))
        lut_res = 2 ** pow_two_res
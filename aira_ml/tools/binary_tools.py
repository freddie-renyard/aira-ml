from distutils.log import warn
from bitstring import BitArray

class BinCompiler:

    @classmethod
    def compile_to_uint(cls, value, n_output, n_radix, compile_to_signed=False):
        """ Compiles the passed value to an unsigned representation.
        """

        scale_factor = 2 ** n_radix
        scaled_val = int(abs(value) * scale_factor)

        bin_str = str(BitArray(uint=scaled_val, length=n_output).bin)

        # Sign extend the output string
        while len(bin_str) < n_output:
            bin_str = "0" + bin_str

        return bin_str

    @classmethod
    def compile_to_signed(cls, value, n_output, n_radix):
        """ Compiles the passed value to a binary two's complement number.
        """

        scale_factor = 2 ** n_radix
        scaled_val = int(value * scale_factor)
        
        bin_str = str(BitArray(int=scaled_val, length=n_output).bin)

        return bin_str

    @classmethod
    def compile_to_float():
        """ Compiles the passed value to a custom floating point representation, 
        according to the following protocol:

        - Sign bit (MSB)
            - 0 for positive, 1 for negative
        - Exponent
            - Unsigned
            - Offset by 2 ^ (n-1), where n is the exponent's bit depth.
        - Mantissa (LSBs)
            - Unsigned and normalised
            - Vestigial bit is omitted
        
        - Zero
            - Representation is true when the exponent is 0. 
            - When this is the case,
        the value of the mantissa is ignored: therefore, no subnormal values are 
        permitted. 
            - Negative and positive zero are equivalent.
        """
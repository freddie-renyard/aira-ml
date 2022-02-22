from distutils.log import warn
from bitstring import BitArray
from numpy import expand_dims
from rsa import sign

class BinCompiler:

    @classmethod
    def compile_to_uint(cls, value, n_output, n_radix, compile_to_signed=False):
        """ Compiles the passed value to an unsigned representation.
        """

        scale_factor = 2 ** n_radix
        scaled_val = round(abs(value) * scale_factor)

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
    def compile_to_float(cls, value, n_mantissa, n_exponent):
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

        if value < 0:
            sign_bit = "1"
        else:
            sign_bit = "0"

        n_full_mantissa = n_mantissa + 1 # Add on the vestigial bit

        norm_val = abs(value)
        signed_exp = 0

        max_mantissa_val = 1 - 1.0 / ((2 ** n_full_mantissa)-1)
        max_abs_exp = 2 ** (n_exponent-1) - 1

        if norm_val < 0.5:
            if norm_val == 0.0:
                norm_val = 0.0
                signed_exp = -max_abs_exp - 1
            else:
                while norm_val < 0.5:
                    norm_val *= 2
                    signed_exp -= 1
        else:
            while norm_val > max_mantissa_val: 
                norm_val /= 2
                signed_exp += 1

        if abs(signed_exp) > max_abs_exp:
            if value >= max_mantissa_val:
                # Clip the value to the maximum positive number
                signed_exp = max_abs_exp
                norm_val = max_mantissa_val
            else:
                # Clip to zero
                signed_exp = -max_abs_exp - 1
                norm_val = 0.0

        # Compile exponent to binary
        offset = max_abs_exp + 1
        offset_exp = signed_exp + offset

        exp_bin = cls.compile_to_uint(offset_exp, n_exponent, 0)
        
        # Set the mantissa to zero if exponent represents zero.
        # Unnecessary for hardware, but makes unit tests easier
        if offset_exp == 0:
            norm_val = 0.0
        
        # Compile the mantissa to binary
        mantissa_bin = cls.compile_to_uint(norm_val, n_full_mantissa, n_full_mantissa)

        # Clip the vestigial bit off
        mantissa_bin = mantissa_bin[1:]
    
        return sign_bit + exp_bin + mantissa_bin
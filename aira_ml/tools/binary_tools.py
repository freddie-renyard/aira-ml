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

        if n_output < 2:
            raise TypeError("ERROR: n_output must be 2 bits or greater.")
        if type(n_output) != int:
            raise TypeError("ERROR: n_output must be an integer.")

        uint_str = cls.compile_to_uint(value, (n_output-1), n_radix, compile_to_signed=True)

        if value < 0:

            # Invert the binary string.
            flipped_str = ""
            for bit in uint_str:
                if bit == "1":
                    flipped_str += "0"
                elif bit == "0":
                    flipped_str += "1"

            # Sign extend the binary string.
            while len(flipped_str) < n_output:
                flipped_str = "1" + flipped_str
   
            add_one = int(flipped_str, 2) + 1
            flipped_str = str(int("{0:032b}".format(add_one)))
            
            if len(flipped_str) > n_output:
                flipped_str = flipped_str[0:n_output]

            return flipped_str
        else:
            return "0" + uint_str

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
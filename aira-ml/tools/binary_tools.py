class BinCompiler:

    @classmethod
    def compile_to_uint():
        """ Compiles the passed value to an unsigned representation.
        """

    @classmethod
    def compile_to_signed():
        """ Compiles the passed value to a binary two's complement number.
        """

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
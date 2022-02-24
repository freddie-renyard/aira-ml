from aira_ml.tools.binary_tools import BinCompiler
from random import randint
import numpy as np

def generate_rand_bits(bits, radix):

    out_str = ""
    for _ in range(bits):
        out_str += str(randint(0,1))

    if out_str[0] == "1":
        out_num = int(out_str, 2) - (1 << bits)
    else:
        out_num = int(out_str, 2)

    return out_str, out_num / (2 ** radix)
    
def test_unsigned():
    assert BinCompiler.compile_to_uint(0.5, 3, 3) == "100"
    assert BinCompiler.compile_to_uint(1,8,0) == "00000001"
    assert BinCompiler.compile_to_uint(1.125, 4, 3) == "1001"
    assert BinCompiler.compile_to_uint(255, 8, 0) == "11111111"
    assert BinCompiler.compile_to_uint(15.0, 8, 4) == "11110000"
    assert BinCompiler.compile_to_uint(85, 7, 0) == "1010101"
    assert BinCompiler.compile_to_uint(0.0000001, 4, 4) == "0000"
    assert BinCompiler.compile_to_uint(0.1, 1, 1) == "0"
    assert BinCompiler.compile_to_uint(0.74, 2, 2) == "11"

def test_signed():
    
    assert BinCompiler.compile_to_signed(-1,8,0) == "11111111"
    assert BinCompiler.compile_to_signed(-127,8,0) == "10000001"
    assert BinCompiler.compile_to_signed(-128,8,0) == "10000000"
    assert BinCompiler.compile_to_signed(-9,5,0) == "10111"
    assert BinCompiler.compile_to_signed(9,5,0) == "01001"
    assert BinCompiler.compile_to_signed(-1.5,5,2) == "11010"

    # Randomly test a range of conditions
    for i in range(2,32):
        for _ in range(100):
            test_bin, test_val = generate_rand_bits(i, i-1)
            assert BinCompiler.compile_to_signed(test_val,i,i-1) == test_bin
    
    assert BinCompiler.compile_to_signed(-0.5,5,5) == "10000"

def test_floats():

    dynamic_range = 50
    test_vals = np.linspace(-dynamic_range, dynamic_range, 500)

    for val in test_vals:
        out_str = BinCompiler.compile_to_float(val,4,4)
        out_decoded = BinCompiler.decode_custom_float(out_str,4,4)

        norm_error = abs(out_decoded - val) / out_decoded

        print("Input: {} Output {} String {}".format(val, out_decoded, out_str))   

        assert (norm_error < 0.1) == True
            
    assert BinCompiler.compile_to_float(0.5,4,4) == "0"+"1000"+"0000"
    assert BinCompiler.compile_to_float(-0.999,4,4) == "1"+"1001"+"0000"
    assert BinCompiler.compile_to_float(64,4,4) == "0"+"1111"+"0000"
    assert BinCompiler.compile_to_float(-1024.0, 8, 5) == "1" + "11011" + "00000000"
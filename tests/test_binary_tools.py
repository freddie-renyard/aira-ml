from aira_ml.tools.binary_tools import BinCompiler

def test_unsigned():
    assert BinCompiler.compile_to_uint(0.5, 3, 3) == "100"
    assert BinCompiler.compile_to_uint(1,8,0) == "00000001"
    assert BinCompiler.compile_to_uint(1.125, 4, 3) == "1001"
    assert BinCompiler.compile_to_uint(255, 8, 0) == "11111111"
    assert BinCompiler.compile_to_uint(15.0, 8, 4) == "11110000"
    assert BinCompiler.compile_to_uint(85, 7, 0) == "1010101"
    assert BinCompiler.compile_to_uint(0.0000001, 4, 4) == "0000"
    assert BinCompiler.compile_to_uint(0.1, 1, 1) == "0"

def test_signed():
    assert BinCompiler.compile_to_signed(-1,8,0) == "11111111"
from aira_ml.tools.binary_tools import BinCompiler

def test_unsigned():
    assert BinCompiler.compile_to_uint(0.5, 3, 3) == "100"
    assert BinCompiler.compile_to_uint(1,8,0) == "00000001"
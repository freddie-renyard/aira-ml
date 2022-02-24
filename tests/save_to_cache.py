from aira_ml.tools.file_tools import Filetools
from aira_ml.tools.lut_compiler import LUTCompiler

bin_lst = LUTCompiler.compile_sigmoid(3,3,8)

Filetools.save_to_file("sigmoid_lut", bin_lst)
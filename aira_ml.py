from aira_ml.model_compiler import ModelCompiler
import argparse
import os
from aira_ml.tools.aira_exceptions import AiraException
import json

# Runs model synthesis from the command line and invokes relevant shell scripts to build hardware
# on host machine.
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="store")
    args = parser.parse_args()

    server_fp = "./aira_ml/config/server_config.json"
    if not os.path.isfile(server_fp):
        with open("aira_ml/config/server_config.json", "w+") as file:
            json.dump({
                "project_dir": "",
                "vivado_loc": "",
                "project_loc": "",
                "bitstream_loc": ""
            }, file)
        raise AiraException("AIRA: Warning - server not configured. Please enter configuration details in {}".format(server_fp))

    ModelCompiler.compile_tf_model(args.file)
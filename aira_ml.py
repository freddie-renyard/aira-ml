import sys
from aira_ml.model_compiler import ModelCompiler
import argparse

# Runs model synthesis from the command line and invokes relevant shell scripts to build hardware
# on host machine.
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="store")
    args = parser.parse_args()

    ModelCompiler.compile_tf_model(args.file)
    
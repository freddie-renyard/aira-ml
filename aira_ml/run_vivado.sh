#!/bin/bash
# This script is run on the host machine when the remote Vivado host is logged into over SSH.
# 1st argument - path to Vivado executable
# 2nd argument - path to project file (.xpr)

echo "AIRA: Attempting Vivado run..."
source $2

# Start vivado, passing the project filepath as an argument.
vivado -mode tcl -source ./aira_ml/build_aira.tcl -tclargs $3 ./aira_ml/cache $4


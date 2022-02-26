#!/bin/bash
# This script is run on the host machine when the remote Vivado host is logged into over SSH.
# 1st argument - path to Vivado executable
# 2nd argument - path to project file (.xpr)

echo "Attempting Vivado run..."
source $1

# Start vivado, passing the project filepath as an argument.
vivado -mode tcl -tclargs $2

set filepath [lindex $argv 0]
open_project $filepath
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
wait_on_run write_bitstream
exit
# set filepath [lindex $argv ?]

# Add Aira build to vivado project
open_project [lindex $argv 0]
add_files [lindex $argv 1]

# Synthesise and implemnent hardware design
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# Run hardware server
open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
current_hw_device [get_hw_devices xc7k325t_0] # TODO Generalise this.
refresh_hw_device -update_hw_probes false [lindex [get_hw_devices xc7k325t_0]]
set_property PROGRAM.FILE {[lindex $argv 2]}

exit 
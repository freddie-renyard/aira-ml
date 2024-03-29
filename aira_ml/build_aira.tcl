# set filepath [lindex $argv ?]

# Add Aira build to vivado project
open_project [lindex $argv 0]

cd aira_ml/cache
set source_files [exec ls]

foreach file $source_files {
    puts $file
    remove_files $file
    add_files $file
}

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
current_hw_device [get_hw_devices [lindex $argv 3]]
refresh_hw_device -update_hw_probes false [lindex [get_hw_devices [lindex $argv 3]]]
set_property PROGRAM.FILE [lindex $argv 2] [get_hw_devices [lindex $argv 3]]
program_hw_devices [get_hw_devices [lindex $argv 3]]

exit 
# set filepath [lindex $argv ?]
open_project [lindex $argv 0]
add_files $1
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
wait_on_run write_bitstream

exit 
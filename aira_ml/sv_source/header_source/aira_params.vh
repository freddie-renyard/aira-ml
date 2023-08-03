
// Module Input and Output parameters
parameter N_MODULE_INPUT = <n_mod_in>,
N_MODULE_OUTPUT = <n_mod_out>,
N_MODULE_INPUT_PORTS = <n_mod_in_ports>,
N_MODULE_OUTPUT_PORTS = <n_mod_out_ports>,
N_MODULE_INPUT_LEN = <n_mod_in_len>,
N_MODULE_OUTPUT_LEN = <n_mod_out_len>,
N_MODULE_INPUT_ADDR = $clog2(N_MODULE_INPUT_LEN),
N_MODULE_OUTPUT_ADDR = $clog2(N_MODULE_OUTPUT_LEN);

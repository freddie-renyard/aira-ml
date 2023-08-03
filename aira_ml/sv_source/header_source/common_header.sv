// Layer <i> Common Parameters 
parameter LAYER_ID_<i> = <i>,
N_MAN_INPUT_<i> = <n_man_input>,
N_EXP_INPUT_<i> = <n_exp_input>,
N_INPUT_<i> = <n_input>,
N_INPUT_PORTS_<i> = <n_input_ports>,
INPUT_LEN_<i> = <input_len>,
N_INPUT_ADDR_<i> = $clog2(INPUT_LEN_<i>),

N_MAN_WEIGHT_<i> = <n_man_weight>,
N_EXP_WEIGHT_<i> = <n_exp_weight>,

N_MAN_OUT_<i> = <n_man_out>,
N_EXP_OUT_<i> = <n_exp_out>,
N_OUTPUT_<i> = <n_output>,
N_OUTPUT_PORTS_<i> = <n_output_ports>,
OUTPUT_LEN_<i> = <output_len>,
N_OUTPUT_ADDR_<i> = $clog2(OUTPUT_LEN_<i>),

N_OVERFLOW_<i> = <n_overflow>,
MULT_EXTRA_<i> = <mult_extra>,
ACT_FUNC_<i> = <act_code>;

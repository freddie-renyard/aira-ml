
  
    dense_top #(
        .PRE_NEURON_NUM(PRE_NEURON_NUM_<i>),
        .POST_NEURON_NUM(POST_NEURON_NUM_<i>),
        .N_MAN_INPUT(N_MAN_INPUT_<i>),
        .N_EXP_INPUT(N_EXP_INPUT_<i>),
        .N_MAN_WEIGHT(N_MAN_WEIGHT_<i>),
        .N_EXP_WEIGHT(N_EXP_WEIGHT_<i>),
        .N_MAN_OUT(N_MAN_OUT_<i>),
        .N_EXP_OUT(N_EXP_OUT_<i>),
        .N_OVERFLOW(N_OVERFLOW_<i>),
        .MULT_EXTRA(MULT_EXTRA_<i>),
        .MEM_DEPTH(MEM_DEPTH_<i>), 
        .ACT_FUNC(ACT_FUNC_<i>),
        .N_ADDR_DELTA(N_ADDR_DELTA_<i>),
        .FILE_ID(FILE_ID_<i>)
    ) dense_<i> (
        .clk(clk),
        .rst(rst),
        
        // Input ports
        .i_data(i_data_<i>),
        .i_d_valid(i_d_valid_<i>),
        .o_stall(o_stall_<i>),
        
        // Output
        .o_data(o_data_<i>),
        .o_d_valid(o_d_valid_<i>),
        .i_stall(i_stall_<i>)
    );
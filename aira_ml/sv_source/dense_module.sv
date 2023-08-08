

    dense_new_top #(
        .LAYER_ID(LAYER_ID_<i>),
        .N_MAN_INPUT(N_MAN_INPUT_<i>),
        .N_EXP_INPUT(N_EXP_INPUT_<i>),
        .N_MAN_WEIGHT(N_MAN_WEIGHT_<i>),
        .N_EXP_WEIGHT(N_EXP_WEIGHT_<i>),
        .N_MAN_OUT(N_MAN_OUT_<i>),
        .N_EXP_OUT(N_EXP_OUT_<i>),
        .N_OVERFLOW(N_OVERFLOW_<i>),
        .MULT_EXTRA(MULT_EXTRA_<i>),
        .ACT_FUNC(ACT_FUNC_<i>),
        .N_INPUT_PORTS(N_INPUT_PORTS_<i>),
        .N_OUTPUT_PORTS(N_OUTPUT_PORTS_<i>),
        .N_INPUT_ADDR(N_INPUT_ADDR_<i>),
        .N_OUTPUT_ADDR(N_OUTPUT_ADDR_<i>),
        .PRE_NEURON_NUM(INPUT_LEN_<i>),
        .POST_NEURON_NUM(OUTPUT_LEN_<i>),
        .THREADS(THREADS_<i>),
        .N_ADDR_DELTA(N_ADDR_DELTA_<i>),
        .MEM_DEPTHS(MEM_DEPTHS_<i>),
        .LUT_DEPTH(LUT_DEPTH_<i>)
    ) layer_<i> (
        .clk(clk),
        .rst(rst),
        
        // Input ports
        .i_data(i_data_<i>),
        .i_d_addr(i_d_addr_<i>),
        .i_d_valid(i_d_valid_<i>),
        .o_stall(o_stall_<i>),
        
        // Output
        .o_data(o_data_<i>),
        .o_d_addr(o_d_addr_<i>),
        .o_d_valid(o_d_valid_<i>),
        .i_stall(i_stall_<i>)
    );
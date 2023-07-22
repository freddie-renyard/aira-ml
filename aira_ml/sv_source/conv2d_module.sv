
conv2d_top #(
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
        .N_CHAN(N_CHAN_<i>),
        .N_THREAD_CHAN(N_THREAD_CHAN_<i>),
        .N_FILTER(N_FILTER_<i>),
        .FILTER_DIM(FILTER_DIM_<i>),
        .N_COL(N_COL_<i>),
        .N_ROW(N_ROW_<i>),
        .N_THREAD_ROWCOL(N_THREAD_ROWCOL_<i>),
        .N_THREAD_FILTER(N_THREAD_FILTER_<i>),
        .ENTRY_PTRS(ENTRY_PTRS_<i>),
        .MAX_POOL(MAX_POOL_<i>)
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
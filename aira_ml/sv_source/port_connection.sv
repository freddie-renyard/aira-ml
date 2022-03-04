

    // Connection from object <i_pre> to <i_post>
    assign i_data_<i_post> = o_data_<i_pre>;
    assign i_d_valid_<i_post> = o_d_valid_<i_pre>;
    assign o_stall_<i_post> = i_stall_<i_pre>;
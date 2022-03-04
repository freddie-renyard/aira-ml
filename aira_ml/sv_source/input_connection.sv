

    // Connection from top file input port to <i_post>
    assign i_data_<i_post> = i_data;
    assign i_d_valid_<i_post> = i_d_valid;
    assign o_stall = o_stall_<i_post>;
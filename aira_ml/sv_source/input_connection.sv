

    // Connection from top file input port to <i_post>
    always_comb begin
        i_data_<i_post>     = i_data;
        i_d_addr_<i_post>   = i_d_addr;
        i_d_valid_<i_post>  = i_d_valid;
        o_stall             = o_stall_<i_post>;
    end


    // Connection from object <i_pre> to <i_post>
    always_comb begin
        i_data_<i_post>     = o_data_<i_pre>;
        i_d_addr_<i_post>   = o_d_addr_<i_pre>;
        i_d_valid_<i_post>  = o_d_valid_<i_pre>;
        i_stall_<i_pre>     = o_stall_<i_post>;
    end
    
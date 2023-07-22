

    // Connection from object <i_pre> to top file output port 
    always_comb begin
        o_data = o_data_<i_pre>;
        o_d_addr = o_d_addr_<i_pre>;
        o_d_valid = o_d_valid_<i_pre>;
        i_stall_<i_pre> = i_stall;
    end
    

    
    // Layer <i> Wires
    logic [N_INPUT_<i>-1:0]         i_data_<i>      [N_INPUT_PORTS_<i>-1:0];
    logic [N_INPUT_ADDR_<i>-1:0]    i_d_addr_<i>    [N_INPUT_PORTS_<i>-1:0];
    logic i_d_valid_<i>;
    logic o_stall_<i>;
    
    logic [N_OUTPUT_<i>-1:0]        o_data_<i>      [N_OUTPUT_PORTS_<i>-1:0];
    logic [N_OUTPUT_ADDR_<i>-1:0]   o_d_addr_<i>    [N_OUTPUT_PORTS_<i>-1:0];
    logic o_d_valid_<i>;
    logic i_stall_<i>;
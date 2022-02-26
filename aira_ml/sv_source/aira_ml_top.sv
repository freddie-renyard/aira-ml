module aira_ml (
        input clk,
        input rst,
        
        // Input ports
        input [N_MODULE_INPUT-1:0] i_data_<i_input>,
        input i_d_valid_<i_input>,
        output o_stall_<i_input>,
        
        // Output ports
        output [N_MODULE_OUTPUT-1:0] o_data_<i_output>,
        output o_d_valid_<i_output>,
        input i_stall_<i_output>
    );
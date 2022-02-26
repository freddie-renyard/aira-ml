module aira_ml (
        input clk,
        input rst,
        
        // Input ports
        input [N_MODULE_INPUT-1:0] i_data,
        input i_d_valid,
        output o_stall,
        
        // Output ports
        output [N_MODULE_OUTPUT-1:0] o_data,
        output o_d_valid,
        input i_stall
    );
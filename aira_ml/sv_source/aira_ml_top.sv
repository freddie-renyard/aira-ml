module aira_ml (
        input clk,
        input rst,
        
        // Input ports
        input [N_MODULE_INPUT-1:0]      i_data      [N_MODULE_INPUT_PORTS-1:0],
        input [N_MODULE_INPUT_ADDR-1:0] i_d_addr    [N_MODULE_INPUT_PORTS-1:0],
        input                           i_d_valid,
        output logic                    o_stall,
        
        // Output ports
        output logic [N_MODULE_OUTPUT-1:0]      o_data [N_MODULE_OUTPUT_PORTS-1:0],
        output logic [N_MODULE_OUTPUT_ADDR-1:0] o_d_addr [N_MODULE_OUTPUT_PORTS-1:0],
        output logic                            o_d_valid,
        input                                   i_stall
    );

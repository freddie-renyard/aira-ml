// Conv2D Parameters
parameter N_INPUT_PORTS_<i> = <n_input_ports>; // Number of input ports.  NB Must be the channel dimensionality of the input
parameter N_CHAN_<i> = <n_chan>;
parameter N_THREAD_CHAN_<i> = <n_thread_chan>; // Hardware only supports full parallelization of this at the moment.
parameter N_FILTER_<i> = <n_filter>;
parameter FILTER_DIM_<i> = <filter_dim>;
parameter MAX_POOL = <max_pool>;
parameter N_COL_<i> = <n_col>;        // Including padded dimension
parameter N_ROW_<i> = <n_row>;        // Including padded dimension
parameter N_THREAD_FILTER_<i> = <n_thread_filter>;
parameter N_THREAD_ROWCOL_<i> = <n_thread_rowcol>;
parameter integer ENTRY_PTRS_<i> [N_THREAD_ROWCOL_<i>-1:0] = '{<entry_ptrs>}; // Format: '{0, 40}; 
parameter integer EXIT_PTRS_<i>  [N_THREAD_ROWCOL_<i>-1:0] = '{<exit_ptrs>};  // Format: '{59, 99};

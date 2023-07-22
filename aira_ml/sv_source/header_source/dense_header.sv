// Dense Layer <i> Parameters 
parameter N_ADDR_DELTA_<i> = <n_delta>,
ACT_FUNC_<i> = <act_code>,
THREADS_<i> = <threads>;

parameter integer MEM_DEPTHS_<i> [THREADS_<i>-1:0] = '{<thread-list>};

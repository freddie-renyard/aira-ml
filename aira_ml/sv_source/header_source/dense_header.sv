// Dense Layer <i> Parameters 
parameter N_ADDR_DELTA_<i> = <n_delta>,
THREADS_<i> = <threads>,
PRE_NEURON_NUM_<i> = <pre_neurons>,
POST_NEURON_NUM_<i> = <post_neurons>;

parameter integer MEM_DEPTHS_<i> [THREADS_<i>-1:0] = '{<thread-list>};

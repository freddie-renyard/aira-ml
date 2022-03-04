from aira_ml.aira_objects import DenseAira
import numpy as np

def test_dense_matrix_compiler():

    # Define a sparse test weight matrix
    weights = np.array([
        [0, 0.43, 0, 0.665, 0, 0, 0.11],
        [1.12, 0, 3, 0, 7, 7, 0],
        [0, 0, 3.2, 0.1, 0, 0, 6],
        [1, 3, 0, 7, 9, 4, 0]
    ])
    weights = np.transpose(weights)

    # Define the biases
    biases = [
        0.1,
        0.2,
        0.3, 
        0.4
    ]

    dense_obj = DenseAira(
        index           = 0,
        weights         = weights,
        biases          = biases, 
        act_name        = 'relu',
        n_input_mantissa= 3,
        n_input_exponent= 3,
        n_weight_mantissa= 3,
        n_weight_exponent= 3,
        n_output_mantissa= 4,
        n_output_exponent= 4,
        n_overflow       = 1,
        mult_extra       = 2
    )

    compiled_memory = dense_obj.compile_mem(weights, biases) 
    print(len(compiled_memory))


if __name__ == "__main__":
    test_dense_matrix_compiler()
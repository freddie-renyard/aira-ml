import numpy as np
from matplotlib import pyplot as plt

class MatrixTools:

    @classmethod
    def plot_histogram(cls, tensor, bins=100):
        """Useful tools for observing the general sparsity and dynamic 
        range of a matrix/tensor.

        NB This method plots the matrix without 0.0 entries.
        """
        tensor = np.array(tensor, dtype=float)

        histogram, bin_edges = np.histogram(tensor[tensor != 0.0], bins=bins)

        # Ensure the bar width is equal.
        bar_width = bin_edges[1]-bin_edges[0] 

        plt.bar(bin_edges[:-1], histogram, width=bar_width, align='edge')
        plt.title('Histogram of Tensor Values')
        plt.show()
    
    @staticmethod
    def sparsify_matrix_simple(matrix, density, threshold=1, verbose=False):
        """Prune the matrix to to a specified
        degree of density using thresholding.
        """

        # Number of entries in the tensor
        matrix_val_num = np.product(np.shape(matrix))

        # The acceptible difference between the desired sparsity value
        # and the realised value. Smaller differences lead to longer
        # execution times, and very small differences may not be able
        # to be realised.
        sparse_error = 0.05

        # Limit the number of loops permitted
        execution_limit = 100
        execs = 0

        current_density = 0.0

        while abs(current_density - density) > sparse_error:
            
            # TODO Optimise this algorithm.
            if current_density < density:
                threshold *= 0.6
            else:
                threshold *= 1.4

            bin_matrix = (np.abs(matrix) > threshold)
            current_density = np.sum(bin_matrix) / matrix_val_num

            if verbose:
                print("Current Density: {}".format(current_density))

            execs += 1
            if execs == execution_limit:
                break

        return np.array(matrix * bin_matrix, dtype=float)

    @staticmethod
    def threshold_matrix(matrix, layer_i, threshold=0.01, verbose=False):
        """Prune the matrix using a specified threshold.
        """

        bin_matrix = (np.abs(matrix) > threshold)

        if verbose:
            matrix_val_num = np.product(np.shape(matrix))
            density = np.sum(bin_matrix) / matrix_val_num

            density_perc = density * 100
            
            print("AIRA: Layer {} matrix density - {:.1f}%".format(layer_i, density_perc))

        return np.array(matrix * bin_matrix, dtype=float)
        
        


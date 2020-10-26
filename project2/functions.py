import numpy as np


def design_matrix(X_data, Y_data, f_data, degree):

    n = len(X_data)
    shuffled_idx = np.random.permutation(n)
    p = int((degree+1) * (degree+2) / 2)

    design_matrix = np.zeros([n,p])
    design_matrix[:,0] = 1.0          #First row is simply 1s

    col_idx = 1
    max_degree = 0
    while col_idx < p:
        max_degree += 1
        for i in range(max_degree +1):
            design_matrix[:,col_idx] = X_data[:]**(max_degree-i)*Y_data[:]**i
            col_idx += 1

    design_matrix = design_matrix[shuffled_idx]
    f_data = f_data[shuffled_idx]
    return design_matrix, f_data

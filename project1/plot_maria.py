import matplotlib.pyplot as plt
import numpy as np
from ols import OLS
import sys
import os


N = sys.argv[1]
sigma = sys.argv[2]
path_to_datasets = "./datasets/"
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(N), "sigma", str(sigma)]) + ".txt"

polynomial = [i for i in range(1,15)]
p = len(polynomial)
R2_boot = np.zeros(p)

R2_k_fold = np.zeros(p)
MSE_boot = np.zeros(p)
MSE_k_fold = np.zeros(p)



solver = OLS()
solver.read_data(filename)
for i in range(len(polynomial)):
    solver.create_design_matrix(polynomial[i])
    solver.split_data()
    solver.bootstrap(10000)
    R2_boot[i], MSE_boot[i] = solver.predict_test()
    solver.k_fold_cross_validation(10)
    R2_k_fold[i], MSE_k_fold[i] = solver.predict_test()




plt.plot(polynomial, MSE_boot, label="Bootstrap")
plt.plot(polynomial, MSE_k_fold, label="k-fold cross validation")

plt.xlabel("polynomial degree")
plt.ylabel("MSE")
plt.title("FrankeFunction with N={}, $\sigma ={}$".format(N, sigma))
plt.grid()
plt.legend()
plt.show()

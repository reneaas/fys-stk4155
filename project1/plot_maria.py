import matplotlib.pyplot as plt
import numpy as np
from ols import OLS
import sys
import os


N = sys.argv[1]
sigma = sys.argv[2]
path_to_datasets = "./datasets/"
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(N), "sigma", str(sigma)]) + ".txt"

polynomial = [i for i in range(20)]
p = len(polynomial)
R2_boot = np.zeros(p)

R2_k_fold = np.zeros(p)
MSE_boot = np.zeros(p)
MSE_k_fold = np.zeros(p)
#bias = np.zeros(p)
#variance = np.zeros(p)


R2_bootstrap = np.zeros(p)
MSE_bootstrap = np.zeros(p)
bias_bootstrap = np.zeros(p)
variance_bootstrap = np.zeros(p)


R2_cross_val = np.zeros(p)
MSE_cross_val = np.zeros(p)
bias_cross_val = np.zeros(p)
variance_cross_val = np.zeros(p)

solver = OLS()
solver.read_data(filename)
for i in range(len(polynomial)):
    solver.create_design_matrix(polynomial[i])
    solver.split_data()
    R2_bootstrap[i], MSE_bootstrap[i], bias_bootstrap[i], variance_bootstrap[i] = solver.bootstrap(100)
    #R2_cross_val[i], MSE_cross_val[i], bias_cross_val[i], variance_cross_val[i] = solver.k_fold_cross_validation_old(100)
    R2_cross_val[i], MSE_cross_val[i] = solver.k_fold_cross_validation(100)

plt.plot(polynomial, MSE_bootstrap, label="MSE (bootstrap)")
plt.plot(polynomial, bias_bootstrap, label = "Bias (bootstrap)")
plt.plot(polynomial, variance_bootstrap, label = "variance (bootstrap)")

plt.xlabel("polynomial degree")
plt.ylabel("MSE")
plt.title("FrankeFunction with N={}, $\sigma ={}$".format(N, sigma))
plt.grid()
plt.legend()
plt.figure()

#print(variance_cross_val)
plt.plot(polynomial, MSE_cross_val, label="MSE (cross-validation)")
#plt.plot(polynomial, bias_cross_val, label = "Bias (cross-validation)")
#plt.plot(polynomial, variance_cross_val, label = "variance (cross-validation)")
plt.xlabel("polynomial degree")
plt.ylabel("MSE")
plt.title("FrankeFunction with N={}, $\sigma ={}$".format(N, sigma))
plt.grid()
plt.legend()

plt.show()

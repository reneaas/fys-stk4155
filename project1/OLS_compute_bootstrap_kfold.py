import matplotlib.pyplot as plt
import numpy as np
from ols import OLS
import sys
import os


N = sys.argv[1]
sigma = sys.argv[2]
p = 20
path_to_datasets = "./datasets/"
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(N), "sigma", str(sigma)]) + ".txt"

polynomial = [i for i in range(p)]
b = 1000
k = 5


R2_bootstrap = np.zeros(p)
MSE_bootstrap = np.zeros(p)
bias_bootstrap = np.zeros(p)
variance_bootstrap = np.zeros(p)


R2_cross_val = np.zeros(p)
MSE_cross_val = np.zeros(p)

solver = OLS()
solver.read_data(filename)
for i in range(len(polynomial)):
    solver.create_design_matrix(polynomial[i])
    solver.split_data()
    R2_bootstrap[i], MSE_bootstrap[i], bias_bootstrap[i], variance_bootstrap[i] = solver.bootstrap(b)
    R2_cross_val[i], MSE_cross_val[i] = solver.k_fold_cross_validation(k)

save_path = "./results/FrankeFunction/OLS/" + "_".join(["N", str(N), "Sigma", str(sigma)]) + "/Data/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.save(save_path + "R2_boot_" + str(b) + ".npy", R2_bootstrap)
np.save(save_path + "MSE_boot_" + str(b) + ".npy", MSE_bootstrap)
np.save(save_path + "Bias_boot_" + str(b) + ".npy", bias_bootstrap)
np.save(save_path + "Variance_boot_" + str(b) + ".npy", variance_bootstrap)

np.save(save_path + "R2_k_" + str(k) + ".npy", R2_cross_val)
np.save(save_path + "MSE_k_" + str(k) + ".npy", MSE_cross_val)

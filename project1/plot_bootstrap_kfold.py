import matplotlib.pyplot as plt
import numpy as np
from ols import OLS
import sys
import os


N = sys.argv[1]
sigma = sys.argv[2]
path_to_datasets = "./datasets/"
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(N), "sigma", str(sigma)]) + ".txt"

polynomial = [i for i in range(8)]
p = len(polynomial)
b = 1000
k = 10


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


path = "./results/FrankeFunction/OLS/" + "_".join(["N", str(N),"Sigma",str(sigma)]) +"/"

if not os.path.exists(path):
    os.makedirs(path)

plot_name = path + "Bootstrap_MaxPdeg_" + str(p) + ".pdf"

plt.plot(polynomial, MSE_bootstrap, label="MSE")
plt.plot(polynomial, bias_bootstrap, label = "Bias")
plt.plot(polynomial, variance_bootstrap, label = "variance")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("Statistical values from Bootstrap w/ %i Re-samples" % b)
plt.grid()
plt.legend()
plt.figure()
plt.savefig(plot_name)

plot_name = path + "MSE_Compare_MaxPdeg_" + str(p) + ".pdf"

plt.plot(polynomial, MSE_bootstrap, label="Bootstrap w/ %i Re-samples" % b)
plt.plot(polynomial, MSE_cross_val, label="%i-fold Cross-Validation" % k)
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("MSE values from Bootstrap and K-fold")
plt.grid()
plt.legend()
plt.savefig(plot_name)
plt.show()

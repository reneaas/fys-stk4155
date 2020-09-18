import matplotlib.pyplot as plt
import numpy as np
from ols import OLS
import sys
import os


N = sys.argv[1]
sigma = sys.argv[2]
path_to_datasets = "./datasets/"
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(N), "sigma", str(sigma)]) + ".txt"

polynomial = [i for i in range(1,20)]
p = len(polynomial)
R2_train = np.zeros(p)
MSE_train = np.zeros(p)
R2_test = np.zeros(p)
MSE_test = np.zeros(p)
bias = np.zeros(p)
variance = np.zeros(p)

solver = OLS()
solver.read_data(filename)


for i in range(len(polynomial)):
    solver.create_design_matrix(polynomial[i])
    solver.split_data()
    #solver.bootstrap(100)
    solver.train()
    R2_train[i], MSE_train[i] = solver.predict_train()
    R2_test[i], MSE_test[i] = solver.predict_test()
    bias[i], variance[i] = solver.compute_bias_variance()

plt.plot(polynomial, MSE_train, label="train")
plt.plot(polynomial, MSE_test, label="test")
plt.plot(polynomial, bias, "-v", label="bias", color="m")
plt.xlabel("polynomial degree")
plt.ylabel("MSE")
plt.title("FrankeFunction with N={}, $\sigma ={}$".format(N, sigma))
plt.legend()
plt.show()

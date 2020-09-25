from ridge import Ridge
from lasso import Lasso
from ols import OLS
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
plt.rc("text", usetex=True)

N = 1000 #Number of datapoints
sigma = 0.1
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.

filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(N), "sigma", str(sigma)]) + ".txt"

method = sys.argv[1]
Polynomial_degrees = [i for i in range(20)]
P = len(Polynomial_degrees)
b = 100

if method == "Ridge":

    path_to_plot = "./results/FrankeFunction/Ridge/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)

    Lambdas = [1/10**i for i in range(-5,10)]
    L = len(Lambdas)

    MSE_Test_Ridge = np.zeros([L,P])
    R2_Test_Ridge = np.zeros([L,P])


    solver = Ridge()
    solver.read_data(filename)
    for i in range(P):
        print("Polydeg = %i" % Polynomial_degrees[i])
        solver.create_design_matrix(Polynomial_degrees[i])
        solver.split_data()
        for j in range(L):
            solver.Lambda = Lambdas[j]

            R2, MSE, bias, variance = solver.bootstrap(b)
            MSE_Test_Ridge[j,i] = MSE
            R2_Test_Ridge[j,i] = R2

    np.save("MSE_Ridge_franke_boot_" + str(b) + ".npy", MSE_Test_Ridge)
    np.save("R2_Ridge_franke_boot_" + str(b) + ".npy", R2_Test_Ridge)



if method == "Lasso":

    path_to_plot = "./results/FrankeFunction/Lasso/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)

    Lambdas = [1/10**i for i in range(-1,20)]
    L = len(Lambdas)

    MSE_Test_Lasso = np.zeros([L,P])
    R2_Test_Lasso = np.zeros([L,P])

    solver = Lasso()
    solver.read_data(filename)
    for i in range(P):
        print("Polydeg = %i" % Polynomial_degrees[i])
        solver.create_design_matrix(Polynomial_degrees[i])
        solver.split_data()
        for j in range(L):
            solver.Lambda = Lambdas[j]
            R2, MSE, bias, variance = solver.bootstrap(b)
            MSE_Test_Lasso[j,i] = MSE
            R2_Test_Lasso[j,i] = R2


    np.save("MSE_lasso_franke_boot_" + str(b) + ".npy", MSE_Test_Lasso)
    np.save("R2_lasso_franke_boot_" + str(b) + ".npy", R2_Test_Lasso)

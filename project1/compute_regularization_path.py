from ridge import Ridge
from lasso import Lasso
from ols import OLS
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
plt.rc("text", usetex=True)

N = 1000 #Number of datapoints
sigma = 1.0
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.

filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(N), "sigma", str(sigma)]) + ".txt"

method = sys.argv[1]
Polynomial_degrees = [i for i in range(12)]
P = len(Polynomial_degrees)

if method == "Ridge":

    path_to_plot = "./results/FrankeFunction/Ridge/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)

    Lambdas = [1/10**i for i in range(-5,5)]

    L = len(Lambdas)
    MSE_Test_Ridge = np.zeros([L,P])


    solver = Ridge()
    solver.read_data(filename)
    for i in range(P):
        solver.create_design_matrix(Polynomial_degrees[i])
        solver.split_data()
        for j in range(L):
            solver.Lambda = Lambdas[j]
            solver.train()
            R2, MSE = solver.predict_test()
            MSE_Test_Ridge[j,i] = MSE

    idx_L, idx_P = np.where(MSE_Test_Ridge == np.min(MSE_Test_Ridge))
    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    plot_name = path_to_plot + "Regularization_Path.pdf"

    plt.contourf(P_deg, Lam, MSE_Test_Ridge, cmap = "inferno", levels=20)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.title("MSE Ridge")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Lambda")
    plt.colorbar()
    plt.savefig(plot_name)




if method == "Lasso":

    path_to_plot = "./results/FrankeFunction/Lasso/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)

    Lambdas = [1/10**i for i in range(-1,6)]
    L = len(Lambdas)
    MSE_Test_Lasso = np.zeros([L,P])

    solver = Lasso()
    solver.read_data(filename)
    for i in range(P):
        solver.create_design_matrix(Polynomial_degrees[i])
        solver.split_data()
        for j in range(L):
            solver.Lambda = Lambdas[j]
            solver.train()
            R2, MSE = solver.predict_test()
            MSE_Test_Lasso[j,i] = MSE


    plot_name = path_to_plot + "Regularization_Path.pdf"

    idx_L, idx_P = np.where(MSE_Test_Lasso == np.min(MSE_Test_Lasso))
    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    plt.contourf(P_deg, Lam, MSE_Test_Lasso, cmap = "inferno", levels=20)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.title("MSE Lasso")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Lambda")
    plt.colorbar()
    plt.savefig(plot_name)

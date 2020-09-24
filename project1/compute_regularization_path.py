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
Polynomial_degrees = [i for i in range(22)]
P = len(Polynomial_degrees)
b = 1000

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
            solver.train()
            R2, MSE = solver.predict_test()
            MSE_Test_Ridge[j,i] = MSE
            R2_Test_Ridge[j,i] = R2

    idx_L, idx_P = np.where(MSE_Test_Ridge == np.min(MSE_Test_Ridge))
    R2_idx_L, R2_idx_P = np.where(R2_Test_Ridge == np.max(R2_Test_Ridge))

    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    plot_name = path_to_plot + "MSE_Regularization_Path.pdf"

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, MSE_Test_Ridge, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.text(Polynomial_degrees[idx_P[0]] + 0.3,Lambdas[idx_L[0]] + 0.3, "Max. MSE" + r"($\lambda, p$)", color = "w")
    plt.title("MSE Regularization path - Ridge", fontsize=font_size)
    plt.xlabel("Polynomial Degree", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size-2)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="MSE", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()

    plot_name = path_to_plot + "R2_Regularization_Path.pdf"

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, R2_Test_Ridge, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[R2_idx_P[0]],Lambdas[R2_idx_L[0]], "k+")
    plt.text(Polynomial_degrees[R2_idx_P[0]] + 0.3,Lambdas[R2_idx_L[0]] + 0.3, "Max. R2" + r"($\lambda, p$)", color = "k")
    plt.title("R2 Regularization path - Ridge", fontsize=font_size)
    plt.xlabel("Polynomial Degree", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size-2)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="R2", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()




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
            solver.train()
            R2, MSE = solver.predict_test()
            MSE_Test_Lasso[j,i] = MSE
            R2_Test_Lasso[j,i] = R2


    plot_name = path_to_plot + "MSE_Regularization_Path.pdf"

    idx_L, idx_P = np.where(MSE_Test_Lasso == np.min(MSE_Test_Lasso))
    R2_idx_L, R2_idx_P = np.where(R2_Test_Lasso == np.max(R2_Test_Lasso))

    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, MSE_Test_Lasso, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.text(Polynomial_degrees[idx_P[0]] + 0.3,Lambdas[idx_L[0]] + 0.3, "Max. MSE" + r"($\lambda, p$)", color = "w")
    plt.title("MSE Regularization path - Lasso", fontsize=font_size)
    plt.xlabel("Polynomial Degree", fontsize=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", fontsize=font_size-4)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="MSE", size=font_size)
    cb.ax.tick_params(labelsize=tick_size)
    plt.savefig(plot_name)
    plt.close()

    plot_name = path_to_plot + "R2_Regularization_Path.pdf"

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, R2_Test_Lasso, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[R2_idx_P[0]],Lambdas[R2_idx_L[0]], "k+")
    plt.text(Polynomial_degrees[R2_idx_P[0]] + 0.3,Lambdas[R2_idx_L[0]] + 0.3, "Max. R2" + r"($\lambda, p$)", color = "k")
    plt.title("R2 Regularization path - Lasso", fontsize=font_size)
    plt.xlabel("Polynomial Degree", fontsize=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", fontsize=font_size-4)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="R2", size=font_size)
    cb.ax.tick_params(labelsize=tick_size)
    plt.savefig(plot_name)
    plt.close()

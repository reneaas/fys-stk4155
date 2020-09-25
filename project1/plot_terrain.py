from ridge import Ridge
from lasso import Lasso
from ols import OLS
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
plt.rc("text", usetex=True)


filename = "./datasets/TerrainFiles/terrain_data.txt"
#filename = "./datasets/frankefunction_dataset_N_1000_sigma_0.1.txt"

method = sys.argv[1]
Polynomial_degrees = [i for i in range(24)]
P = len(Polynomial_degrees)
k = 10

if method == "Ridge":

    path_to_plot = "./results/TerrainData/Ridge/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)

    Lambdas = [1/10**i for i in range(-2,7)]

    L = len(Lambdas)
    MSE_Test_Ridge = np.zeros([L,P])
    R2_Test_Ridge = np.zeros([L,P])

    solver = Ridge()
    solver.read_data(filename)
    for i in range(P):
        print("p = ", i)
        solver.create_design_matrix(Polynomial_degrees[i])
        solver.split_data()
        for j in range(L):
            solver.Lambda = Lambdas[j]
            #solver.train()
            #R2, MSE = solver.predict_test()
            R2, MSE = solver.k_fold_cross_validation(k)
            MSE_Test_Ridge[j,i] = MSE
            R2_Test_Ridge[j,i] = R2

    idx_L, idx_P = np.where(MSE_Test_Ridge == np.min(MSE_Test_Ridge))
    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    plot_name = path_to_plot + "Regularization_Path_MSE.pdf"

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, MSE_Test_Ridge, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.title("Regularization path - Ridge", fontsize=font_size)
    plt.xlabel("Polynomial Degree", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="MSE", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()

    idx_L, idx_P = np.where(R2_Test_Ridge == np.max(R2_Test_Ridge))
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    plot_name = path_to_plot + "Regularization_Path_R2.pdf"

    plt.contourf(P_deg, Lam, R2_Test_Ridge, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "k+")
    plt.title("Regularization path - Ridge", fontsize=font_size)
    plt.xlabel("Polynomial Degree", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label=r"$R^2$", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()



if method == "Lasso":

    path_to_plot = "./results/TerrainData/Lasso/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)

    Lambdas = [1/10**i for i in range(-1,6)]
    L = len(Lambdas)
    MSE_Test_Lasso = np.zeros([L,P])
    R2_Test_Ridge = np.zeros([L,P])

    solver = Lasso()
    solver.read_data(filename)
    for i in range(P):
        solver.create_design_matrix(Polynomial_degrees[i])
        solver.split_data()
        for j in range(L):
            solver.Lambda = Lambdas[j]
            #solver.train()
            #R2, MSE = solver.predict_test()
            R2, MSE, variance, bias = solver.bootstrap(B)
            MSE_Test_Lasso[j,i] = MSE
            R2_Test_Ridge[j,i] = R2

    plot_name = path_to_plot + "Regularization_Path.pdf"

    idx_L, idx_P = np.where(MSE_Test_Lasso == np.min(MSE_Test_Lasso))
    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, MSE_Test_Lasso, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.title("Regularization path - Lasso", fontsize=font_size)
    plt.xlabel("Polynomial Degree", fontsize=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", fontsize=font_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="MSE", size=font_size)
    cb.ax.tick_params(labelsize=tick_size)
    plt.savefig(plot_name)
    plt.close()


    idx_L, idx_P = np.where(R2_Test_Ridge == np.min(R2_Test_Ridge))
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    plot_name = path_to_plot + "Regularization_Path_R2.pdf"

    plt.contourf(P_deg, Lam, R2_Test_Ridge, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.title("Regularization path - Ridge", fontsize=font_size)
    plt.xlabel("Polynomial Degree", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label=r"$R^2$", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()

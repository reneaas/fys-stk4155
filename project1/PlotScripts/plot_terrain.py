import matplotlib.pyplot as plt
import sys
import numpy as np
import os
plt.rc("text", usetex=True)
method = sys.argv[1]

if method == "Ridge":
    R2_Test_Ridge = np.load("../results/TerrainData/Ridge/Data/R2_Test_Ridge_Terrain.npy")
    MSE_Test_Ridge = np.load("../results/TerrainData/Ridge/Data/MSE_Test_Ridge_Terrain.npy")

    path_to_plot = "../results/TerrainData/Ridge/Plots/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)

    Polynomial_degrees = [i for i in range(24)]
    Lambdas = [1/10**i for i in range(-2,7)]
    L = len(Lambdas)
    P = len(Polynomial_degrees)


    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)
    idx_L, idx_P = np.where(MSE_Test_Ridge == np.min(MSE_Test_Ridge))
    print("Ridge; min MSE ", np.min(MSE_Test_Ridge))
    print("Ridge; argmin lambda = ", Lambdas[idx_L[0]])
    print("Ridge; argmin p = ", Polynomial_degrees[idx_P[0]])


    plot_name = path_to_plot + "Regularization_Path_MSE.pdf"

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, MSE_Test_Ridge, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.text(Polynomial_degrees[idx_P[0]] - 6, Lambdas[idx_L[0]] + 0.3, "min MSE" + r"$(d, \lambda)$", color = "w", size=14)
    plt.xlabel(r"$d$", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="MSE", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()

    idx_L, idx_P = np.where(R2_Test_Ridge == np.max(R2_Test_Ridge))
    print("Ridge; max R2 = ", np.max(R2_Test_Ridge))
    print("Ridge; argmax lambda = ", Lambdas[idx_L[0]])
    print("Ridge; argmax d = ", Polynomial_degrees[idx_P[0]])
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)
    print("Ridge; min MSE = ", np.min(MSE_Test_Ridge))
    plot_name = path_to_plot + "Regularization_Path_R2.pdf"

    plt.contourf(P_deg, Lam, R2_Test_Ridge, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "k+")
    plt.text(Polynomial_degrees[idx_P[0]] - 6, Lambdas[idx_L[0]] + 0.3, "max " + r"$R^2(d,\lambda)$", color = "k", size=14)
    plt.xlabel(r"$d$", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label=r"$R^2$", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()


if method == "Lasso":

    path_to_plot = "../results/TerrainData/Lasso/Plots/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)

    load_path = "../results/TerrainData/Lasso/Data/"

    Polynomial_degrees = [i for i in range(24)]
    Lambdas = [1/10**i for i in range(-1,6)]
    L = len(Lambdas)
    P = len(Polynomial_degrees)

    R2_Test_Lasso = np.load(load_path + "R2_Test_Lasso_Terrain.npy")
    MSE_Test_Lasso = np.load(load_path +"MSE_Test_Lasso_Terrain.npy")

    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)
    idx_L, idx_P = np.where(MSE_Test_Lasso == np.min(MSE_Test_Lasso))
    print("Lasso; min MSE ", np.min(MSE_Test_Lasso))
    print("Lasso; argmin lambda = ", Lambdas[idx_L[0]])
    print("Lasso; argmin d = ", Polynomial_degrees[idx_P[0]])

    idx_L, idx_P = np.where(R2_Test_Lasso == np.max(R2_Test_Lasso))
    print("Lasso; max R2 = ", np.max(R2_Test_Lasso))
    print("Lasso; argmax lambda = ", Lambdas[idx_L[0]])
    print("Lasso; argmax d = ", Polynomial_degrees[idx_P[0]])
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    plot_name = path_to_plot + "Regularization_Path.pdf"

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, MSE_Test_Lasso, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.text(Polynomial_degrees[idx_P[0]] - 5, Lambdas[idx_L[0]] + 0.3, "min MSE" + r"$(d, \lambda)$", color = "k")
    plt.xlabel("Polynomial Degree", fontsize=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", fontsize=font_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="MSE", size=font_size)
    cb.ax.tick_params(labelsize=tick_size)
    plt.savefig(plot_name)
    plt.close()


    idx_L, idx_P = np.where(R2_Test_Lasso == np.max(R2_Test_Lasso))
    print("Lasso; max R2 = ", np.max(R2_Test_Lasso))
    print("Lasso; argmax lambda = ", Lambdas[idx_L[0]])
    print("Lasso; argmax p = ", Polynomial_degrees[idx_P[0]])
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    plot_name = path_to_plot + "Regularization_Path_R2.pdf"

    plt.contourf(P_deg, Lam, R2_Test_Lasso, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[idx_P[0]],Lambdas[idx_L[0]], "w+")
    plt.text(Polynomial_degrees[idx_P[0]] - 5, Lambdas[idx_L[0]] + 0.3, "max " + r"$R^2(\lambda, p)$", color = "k")
    plt.xlabel("Polynomial Degree", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label=r"$R^2$", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()


if method == "OLS":
    R2 = np.load("../results/TerrainData/OLS/Data/R2_test_OLS_terrain.npy")
    MSE = np.load("../results/TerrainData/OLS/Data/MSE_test_OLS_terrain.npy")

    P = [i for i in range(len(R2))]

    plt.plot(P, R2)
    plt.xlabel(r"$d$")
    plt.ylabel(r"$R^2$")
    plt.figure()
    plt.plot(P, MSE)
    plt.xlabel(r"$d$")
    plt.ylabel("MSE")
    plt.show()

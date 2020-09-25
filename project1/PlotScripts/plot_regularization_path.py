import matplotlib.pyplot as plt
import sys
import numpy as np
import os
plt.rc("text", usetex=True)


Polynomial_degrees = [i for i in range(20)]
P = len(Polynomial_degrees)

method = sys.argv[1]
b = sys.argv[2]

if method == "Ridge":

    load_path = "../results/FrankeFunction/Ridge/Data/"

    MSE = np.load(load_path + "MSE_Ridge_franke_boot_" + str(b) + ".npy")

    path_to_plot = "../results/FrankeFunction/Ridge/Plots/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)
    plot_name = path_to_plot + "MSE_Regularization_Path_Boot_" + str(b) + ".pdf"

    Lambdas = [1/10**i for i in range(-5,10)]
    L = len(Lambdas)

    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    M_idx_L, M_idx_P = np.where(MSE == np.min(MSE))
    print("Ridge; min MSE = ", np.min(MSE))
    print("Best p = ",Polynomial_degrees[M_idx_P[0]], "Best lambda = ",Lambdas[M_idx_L[0]])

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, MSE, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[M_idx_P[0]],Lambdas[M_idx_L[0]], "w+")
    plt.text(Polynomial_degrees[M_idx_P[0]] + 0.3,Lambdas[M_idx_L[0]] + 0.3, "Min.MSE" + r"($\lambda, p$)", color = "w")
    plt.xlabel("Polynomial Degree", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size-2)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="MSE", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()


    R2 = np.load(load_path + "R2_Ridge_franke_boot_" + str(b) + ".npy")
    R2_idx_L, R2_idx_P = np.where(R2 == np.max(R2))
    plot_name = path_to_plot + "R2_Regularization_Path_Boot_" + str(b) + ".pdf"

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, R2, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[R2_idx_P[0]],Lambdas[R2_idx_L[0]], "k+")
    plt.text(Polynomial_degrees[R2_idx_P[0]] + 0.3,Lambdas[R2_idx_L[0]] + 0.3, "Max." + r"$R^2(\lambda, p$)", color = "k")
    plt.xlabel("Polynomial Degree", size=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size-2)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label=r"$R^2$", size=14)
    cb.ax.tick_params(labelsize=14)
    plt.savefig(plot_name)
    plt.close()




if method == "Lasso":

    load_path = "../results/FrankeFunction/Lasso/Data/"

    MSE = np.load(load_path + "MSE_lasso_franke_boot_" + str(b) + ".npy")
    M_idx_L, M_idx_P = np.where(MSE == np.min(MSE))

    path_to_plot = "../results/FrankeFunction/Lasso/Plots/"
    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)
    plot_name = path_to_plot + "MSE_Regularization_Path_Boot_" + str(b) + ".pdf"



    Lambdas = [1/10**i for i in range(-1,20)]
    L = len(Lambdas)

    Lambdas = np.log10(Lambdas)
    P_deg, Lam = np.meshgrid(Polynomial_degrees, Lambdas)

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, MSE, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[M_idx_P[0]],Lambdas[M_idx_L[0]], "w+")
    plt.text(Polynomial_degrees[M_idx_P[0]] + 0.3, Lambdas[M_idx_L[0]] + 0.3, "Min.MSE" + r"($\lambda, p$)", color = "w")
    plt.xlabel("Polynomial Degree", fontsize=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", fontsize=font_size-4)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="MSE", size=font_size)
    cb.ax.tick_params(labelsize=tick_size)
    plt.savefig(plot_name)
    plt.close()

    print("Lasso; min MSE = ", np.min(MSE))
    print("best p = ", Polynomial_degrees[M_idx_P[0]], "best lambda = ", Lambdas[M_idx_L[0]])

    plot_name = path_to_plot + "R2_Regularization_Path_Boot_" + str(b) + ".pdf"
    R2 = np.load(load_path + "R2_lasso_franke_boot_" + str(b) + ".npy")
    R2_idx_L, R2_idx_P = np.where(R2 == np.max(R2))

    font_size = 14
    tick_size = 14
    plt.contourf(P_deg, Lam, R2, cmap = "inferno", levels=40)
    plt.plot(Polynomial_degrees[R2_idx_P[0]],Lambdas[R2_idx_L[0]], "k+")
    plt.text(Polynomial_degrees[R2_idx_P[0]] + 0.3,Lambdas[R2_idx_L[0]] + 0.3, "Max. " + r"$R^2(\lambda, p$)", color = "k")
    plt.xlabel("Polynomial Degree", fontsize=font_size)
    plt.ylabel(r"$\log_{10}(\lambda)$", fontsize=font_size-4)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label=r"$R^2$", size=14)
    cb.ax.tick_params(labelsize=tick_size)
    plt.savefig(plot_name)
    plt.close()

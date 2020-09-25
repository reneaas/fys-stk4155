import matplotlib.pyplot as plt
import numpy as np
import sys
import os

N = sys.argv[1]
sigma = sys.argv[2]
polynomial = [i for i in range(20)]

path = "../results/FrankeFunction/OLS/" + "_".join(["N", str(N), "Sigma", str(sigma)])

load_path = path + "/Data/"
save_path = path + "/Plots/"

if not os.path.exists(save_path):
    os.makedirs(save_path)


MSE_boot = np.load(load_path + "MSE_boot_1000.npy")
R2_boot = np.load(load_path + "R2_boot_1000.npy")
bias_boot = np.load(load_path + "Bias_boot_1000.npy")
variance_boot = np.load(load_path + "Variance_boot_1000.npy")

MSE_k = np.load(load_path + "MSE_k_5.npy")
R2_k = np.load(load_path + "R2_k_5.npy")

Min_bootstrap_MSE = np.where(MSE_boot == np.min(MSE_boot))
Min_kfold_MSE = np.where(MSE_k == np.min(MSE_k))


plot_name = save_path + "Bootstrap_statvals.pdf"


font_size = 14
tick_size = 14
plt.plot(polynomial, MSE_boot, label="MSE")
plt.plot(polynomial, bias_boot,"--",label = "Bias")
plt.plot(polynomial, variance_boot,"--" ,label = "variance")
plt.xlabel("Polynomial degree", fontsize=font_size)
plt.ylabel("MSE", fontsize=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
plt.title("Statistical values from Bootstrap w/1000 Re-samples")
plt.grid()
plt.legend()
plt.savefig(plot_name)
plt.close()


plot_name = save_path + "MSE_Compare.pdf"

plt.plot(polynomial, MSE_boot, label="Bootstrap w/1000 Re-samples")
plt.plot(polynomial, MSE_boot, "*")
plt.plot(polynomial[int(Min_bootstrap_MSE[0])], MSE_boot[int(Min_bootstrap_MSE[0])], "o")
plt.plot(polynomial, MSE_k, label="5-fold Cross-Validation")
plt.plot(polynomial, MSE_k, "*")
plt.plot(polynomial[int(Min_kfold_MSE[0])], MSE_k[int(Min_kfold_MSE[0])], "o")
plt.xlabel("Polynomial degree", fontsize=font_size)
plt.ylabel("MSE", fontsize=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
plt.title("MSE values from Bootstrap and K-fold")
plt.grid()
plt.legend()
plt.savefig(plot_name)
plt.close()

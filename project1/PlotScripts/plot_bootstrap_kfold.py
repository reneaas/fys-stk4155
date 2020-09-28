import matplotlib.pyplot as plt
import numpy as np
import sys
import os
plt.rc("text", usetex=True)

N = sys.argv[1]
sigma = sys.argv[2]
p = int(sys.argv[3])
polynomial = [i for i in range(p)]

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

Min_bootstrap_MSE = np.where(MSE_boot[:p] == np.min(MSE_boot[:p]))
Min_kfold_MSE = np.where(MSE_k[:p] == np.min(MSE_k[:p]))


plot_name = save_path + "Bootstrap_statvals_maxdeg_" + str(p-1) + ".pdf"


font_size = 16
tick_size = 16
plt.plot(polynomial, MSE_boot[:p], label="MSE")
plt.plot(polynomial, bias_boot[:p],"--",label = "Bias")
plt.plot(polynomial, variance_boot[:p],"--" ,label = "variance")
plt.xlabel(r"$d$", fontsize=font_size)
plt.ylabel("MSE", fontsize=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
plt.grid()
plt.legend(fontsize=font_size)
plt.savefig(plot_name)
plt.close()


plot_name = save_path + "MSE_Compare_maxdeg_" + str(p-1) + ".pdf"

plt.plot(polynomial, MSE_boot[:p], label="Bootstrap w/1000 Re-samples")
plt.plot(polynomial, MSE_boot[:p], "*")
plt.plot(polynomial[int(Min_bootstrap_MSE[0])], MSE_boot[int(Min_bootstrap_MSE[0])], "o")
plt.plot(polynomial, MSE_k[:p], label="5-fold Cross-Validation")
plt.plot(polynomial, MSE_k[:p], "*")
plt.plot(polynomial[int(Min_kfold_MSE[0])], MSE_k[int(Min_kfold_MSE[0])], "o")
plt.xlabel(r"$d$", fontsize=font_size)
plt.ylabel("MSE", fontsize=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
plt.grid()
plt.legend(fontsize=font_size)
plt.savefig(plot_name)
plt.close()

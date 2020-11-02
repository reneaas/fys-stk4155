import numpy as np
import matplotlib.pyplot as plt
import os

path ="./results/LogisticRegression/"
"""
gammas = np.load(path + "LogReg_Gamma.npy")
etas = np.load(path + "LogReg_Eta.npy")
accu = np.load(path + "LogReg_Accuracy.npy")


path_to_plot = "./results/LogisticRegression/Plots/"
if not os.path.exists(path_to_plot):
    os.makedirs(path_to_plot)

accuracy = np.zeros([len(etas),len(gammas)])
accuracy.flat[:] = accu

etas, gammas = np.meshgrid(etas, gammas)
idx_E, idx_G = np.where(accuracy == np.max(accuracy))
print(idx_E, idx_G)


print("LogReg; max accuracy ", np.max(accuracy))

max_eta = 0.01*idx_E[0]
max_gamma = 0.15 + 0.01*idx_G[0]

plot_name = path_to_plot + "LogisticRegression_accuracy_analasys.pdf"
font_size = 12
tick_size = 12
plt.contourf(etas, gammas, accuracy.T, cmap = "inferno", levels=20)
plt.plot(max_eta, max_gamma, "k+")
plt.text(max_eta, max_gamma, "max accuracy" + r"$(\gamma, \eta)$", color = "k", size=font_size)
plt.ylabel(r"$\gamma$", size=font_size)
plt.xlabel(r"$\eta$", size=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
cb = plt.colorbar()
cb.set_label(label="Accuracy", size=font_size)
cb.ax.tick_params(labelsize=font_size)
plt.savefig(plot_name)
plt.close()
"""

path_to_plot = "./results/LogisticRegression/Plots/"
if not os.path.exists(path_to_plot):
    os.makedirs(path_to_plot)

gammas = np.load(path + "LogReg_Gamma_broad.npy")
etas = np.load(path + "LogReg_Eta_broad.npy")
accu = np.load(path + "LogReg_Accuracy_broad.npy")

accuracy = np.zeros([len(etas),len(gammas)])
accuracy.flat[:] = accu

nogrid_etas = np.copy(etas)
nogrid_gammas = np.copy(gammas)
etas, gammas = np.meshgrid(etas, gammas)

idx_E, idx_G = np.where(accuracy == np.max(accuracy))

print("LogReg; max accuracy ", np.max(accuracy))

plot_name = path_to_plot + "LogisticRegression_accuracy_analasys_broad.pdf"
font_size = 12
tick_size = 12
plt.contourf(etas, gammas, accuracy.T, cmap = "inferno", levels=40)
for i in range(len(idx_E)):
    plt.plot(nogrid_etas[idx_E[i]], nogrid_gammas[idx_G[i]], "k+")
    plt.text(nogrid_etas[idx_E[i]], nogrid_gammas[idx_G[i]], "max accuracy" + r"$(\gamma, \eta)$", color = "k", size=font_size)
plt.ylabel(r"$\gamma$", size=font_size)
plt.xlabel(r"$\eta$", size=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
cb = plt.colorbar()
cb.set_label(label="Accuracy", size=font_size)
cb.ax.tick_params(labelsize=font_size)
plt.savefig(plot_name)
plt.close()

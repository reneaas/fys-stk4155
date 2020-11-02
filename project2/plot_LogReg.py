import numpy as np
import matplotlib.pyplot as plt
import os


gammas = np.load("LogReg_Gamma.npy")
etas = np.load("LogReg_Eta.npy")
accu = np.load("LogReg_Accuracy.npy")


path_to_plot = "./results/LogisticRegression/Plots/"
if not os.path.exists(path_to_plot):
    os.makedirs(path_to_plot)

accuracy = np.zeros([11,11])
accuracy.flat[:] = accu

etas, gammas = np.meshgrid(etas, gammas)
idx_E, idx_G = np.where(accuracy == np.max(accuracy))
print(idx_E, idx_G)

print("LogReg; max accuracy ", np.max(accuracy))

max_eta = 0.01*idx_E[0]
max_gamma = 0.15 + 0.01*idx_G[0]
print(max_gamma)

plot_name = path_to_plot + "LogisticRegression_accuracy_analasys.pdf"
font_size = 12
tick_size = 12
plt.contourf(etas, gammas, accuracy, cmap = "inferno", levels=40)
plt.plot(max_eta, max_gamma, "k+")
plt.text(max_eta, max_gamma, "max accuracy = f" + r"$(\gamma, \eta)$", color = "k", size=font_size)
plt.ylabel(r"$\gamma$", size=font_size)
plt.xlabel(r"$\eta$", size=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
cb = plt.colorbar()
cb.set_label(label="Accuracy", size=font_size)
cb.ax.tick_params(labelsize=font_size)
plt.savefig(plot_name)
plt.close()

import matplotlib.pyplot as plt
import sys
import numpy as np
import os
plt.rc("text", usetex=True)

path = "../results/FrankeFunction/OLS/Hasties/"
plot_name = path + "HastieFig.pdf"

if not os.path.exists(path):
    os.makedirs(path)

p = [i for i in range(15)]

MSE_train = np.load(path + "MSE_train.npy")
MSE_test = np.load(path + "MSE_test.npy")

plt.plot(p, MSE_train, label = "In-sample error")
plt.plot(p, MSE_test, label = "Out-of-sample error")

font_size = 14
tick_size = 14


plt.xlabel("Polynomial Degree", size=font_size)
plt.ylabel("MSE", size=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
plt.grid()
plt.legend(fontsize=font_size)
plt.savefig(plot_name)

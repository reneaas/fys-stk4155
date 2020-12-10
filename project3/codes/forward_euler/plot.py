import matplotlib.pyplot as plt
import numpy as np
import os



def exact_1D(x, t):
    """
    Analytical solution of the PDE in the 1D-case.
    """
    return np.sin(np.pi*x)*np.exp(-np.pi**2 * t)

path = "../results/euler/"

dx = [0.1, 0.01]

t = [0.02, 1.]

fontsize = 16
ticksize = 16
for i in dx:
    for j in t:
        x = []
        u = []
        infilename = path + "euler_dx_" + str(i) + "_time_" + str(j) + ".txt"
        with open(infilename, "r") as infile:
            T = float(infile.readline().split()[0])
            print(T)
            lines = infile.readlines()
            for line in lines:
                vals = line.split()
                x.append(float(vals[0]))
                u.append(float(vals[-1]))

        figurename = "euler_dx_" + str(i) + "_time_" + str(j) + ".pdf"

        x = np.array(x)
        u = np.array(u)
        u_analytical = exact_1D(x, T)
        plt.plot(x, u, label="approximation")
        plt.plot(x, u_analytical, label="analytical")
        plt.xticks(size=ticksize)
        plt.yticks(size=ticksize)
        plt.xlabel("x", size=fontsize)
        plt.ylabel("u(x,%.2f)" % T, size=fontsize)
        plt.legend(fontsize=fontsize)
        plt.savefig(path + figurename)
        plt.close()

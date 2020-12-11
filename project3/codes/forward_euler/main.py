import numpy as np
import os
import sys
import matplotlib.pyplot as plt



path = "../results/euler/"
if not os.path.exists(path):
    os.makedirs(path)

"""
dx = [0.1, 0.01]
total_time = [0.02, 1.]
r = 0.5

for i in dx:
    for t in total_time:
        outfilename = "euler_dx_" + str(i) + "_time_" + str(t) + ".txt"
        command = " ".join(["./main.exe", outfilename, str(t), str(i)])
        os.system(command)
        os.system("mv" + " " + outfilename +" "+ path)
"""

filename1 = path + "u_dx_0.1.txt"
filename2 = path + "u_dx_0.01.txt"
filename3 = path + "u_dx_0.01_t_1.2.txt"

def exact_1D(x, t):
    """
    Analytical solution of the PDE in the 1D-case.
    """
    return np.sin(np.pi*x)*np.exp(-np.pi**2 * t)



t = []
x = []

with open(filename3, "r") as infile:
    first_line = infile.readline()
    points = first_line.split()
    timepoints = int(points[0])
    gridpoints = int(points[1])

    u = np.zeros([timepoints, gridpoints])
    for i in range(timepoints):
        for j in range(gridpoints):
            line = infile.readline()
            values = line.split()
            u[i,j] = float(values[-1])
            if i == 0:
                x.append(float(values[1]))
            t.append(float(values[0]))


t = np.array(t[::gridpoints])
x = np.array(x)

X, T = np.meshgrid(x, t)

fontsize = 16
ticksize = 16

fig = plt.figure()
ax = fig.add_subplot(111)

exact = exact_1D(X,T)


rel_err = np.abs(u - exact)/exact


plt.contourf(X, T, rel_err, cmap="inferno", levels=201)


#plt.contourf(X, T, rel_err, cmap="inferno", levels=41)
cbar = plt.colorbar()
cbar.set_label("relative error", size=fontsize)
cbar.ax.tick_params(labelsize=ticksize)
cbar.formatter.set_powerlimits((0,0))
cbar.ax.yaxis.get_offset_text().set_fontsize(14)
cbar.update_ticks()
plt.xticks(size=ticksize)
plt.yticks(size=ticksize)
ax.set_xlabel(r"$x$", size=fontsize)
ax.set_ylabel(r"$t$", size=fontsize)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os
#plt.rc("text", usetex=True)

infilename = "./results/regression/regression_grid_search_nodes_deg_val.txt"


x = []
y = []
r2 = []
with open(infilename, "r") as infile:
    line = infile.readline()
    lines = infile.readlines()

    for line in lines:
        vals = line.split()
        x.append(float(vals[0]))
        y.append(float(vals[1]))
        r2.append(float(vals[2]))


x = np.array(x)
x = np.unique(x)
y = np.array(y)
n_y = np.array(y)
y = np.unique(y)
r2 = np.array(r2)
r2_val = np.zeros([len(x), len(y)])
r2_val.flat[:] = r2[:]

"""

idx = np.where(x <= 7)
x = x[idx]
r2_val = r2_val[idx]

idx = np.where(x >= 3)
x = x[idx]
r2_val = r2_val[idx]
"""


x = [int(i) for i in x]
y = [int(i) for i in y]


sb.set(font_scale=1.25)
heat_map = sb.heatmap(r2_val.T, annot=True, cbar=True, cbar_kws={"label": "$R^2$", "orientation" : "vertical"})
heat_map.set_xlabel("hidden neurons")
heat_map.set_ylabel("polynomial degree")
heat_map.set_xticklabels(x)
heat_map.set_yticklabels(y)
heat_map.xaxis.tick_top()
heat_map.tick_params(length=0)
plt.savefig("NN_regression_heatmap_neurons_degree.pdf")
plt.show()

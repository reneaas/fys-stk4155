import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
#plt.rc("text", usetex=True)

def read_data(filename):
    x = []
    y = []
    z = []
    x_append = x.append
    y_append = y.append
    z_append = z.append
    with open(filename, "r") as infile:
        line = infile.readline()
        words = line.split()
        xlabel = words[0]
        ylabel = words[1]
        lines = infile.readlines()
        for line in lines:
            vals = line.split()
            x_append(float(vals[0]))
            y_append(float(vals[1]))
            z_append(float(vals[2]))

    x = np.array(x)
    x = np.unique(x)
    y = np.array(y)
    y = np.unique(y)
    Z = np.zeros([len(x), len(y)])

    for i in range(len(x)):
        for j in range(len(y)):
            Z[i,j] = z[i*len(y) + j]


    return x, y, Z, xlabel, ylabel


filename_val = "results/classification/grid_search_lamb_gamma_val.txt"
x, y, z_val, xlabel, ylabel = read_data(filename=filename_val)

filename_test = "results/classification/grid_search_lamb_gamma_test.txt"
x, y, z_test, xlabel, ylabel = read_data(filename=filename_test)


def plot_data(x,y, data, xlabel, ylabel, figurename):
    sb.set(font_scale=1.25)
    heat_map = sb.heatmap(data.T, annot=True, cbar=True, cbar_kws={"label" : "Accuracy (%)"}) #cbar_kws={"label": "$", "orientation" : "vertical"})
    heat_map.set_xlabel(xlabel)
    heat_map.set_ylabel(ylabel)
    heat_map.set_xticklabels(x)
    heat_map.set_yticklabels(y)
    plt.show()
    #plt.savefig(figurename)


figurename_val = filename_val.strip(".txt") + ".pdf"
figurename_test = filename_test.strip(".txt") + ".pdf"
print(figurename_val)

xlabel = "$\lambda$"
ylabel = "$\gamma$"
plot_data(x = x, y = y, data = z_val.T, xlabel = xlabel, ylabel = ylabel, figurename=figurename_val)
plot_data(x, y, z_test/z_val, xlabel, ylabel, figurename=figurename_test)

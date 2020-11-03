import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    x = []
    y = []
    z = []
    x_append = x.append
    y_append = y.append
    z_append = z.append
    with open(filename, "r") as infile:
        lines = infile.readlines()
        for line in lines:
            vals = line.split()
            x_append(float(vals[0]))
            y_append(float(vals[1]))
            z_append(float(vals[2]))

    print(x)
    print(y)
    print(z)
    x = np.array(x)
    x = np.unique(x)
    y = np.array(y)
    y = np.unique(y)
    Z = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i,j] = z[i*len(y) + j]

    return x, y, Z


filename = "./results/grid_search_result.txt"
x, y, z = read_data(filename)
print(x)
print(y)
print(z)





def plot_data(x,y,data):
    # plot results
    fontsize=16



    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1, cmap="inferno")

    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.2f}\\%$".format( 100*data[j,i])
            if 100*data[j,i] < 50:
                ax.text(x_val, y_val, c, va='center', ha='center', color="w")
            else:
                ax.text(x_val, y_val, c, va='center', ha='center', color="k")

    # convert axis vaues to to string labels

    y = y[::-1]
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{data\\ size}$',fontsize=fontsize)

    plt.tight_layout()

    plt.show()

plot_data(x, y, z)

import matplotlib.pyplot as plt
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
            Z[j,i] = z[i*len(y) + j]

    return x, y, Z


#filename = "./results/grid_search_learningrate_datasz.txt"
#filename = "./results/grid_search_neurons_epochs.txt"
#filename = "./results/grid_search_regularization_lambda.txt"
x, y, z = read_data(filename)
print(x)
print(y)
print(z)

def plot_data(x, y, data, figurename):
    # plot results
    fontsize=16
    fontsize2=12
    fontsize3=18

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1, cmap="inferno")

    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])
    cbar.ax.tick_params(labelsize=14)

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.2f}\\%$".format( 100*data[j,i])
            if 100*data[j,i] < 50:
                ax.text(x_val, y_val, c, va='center', ha='center', color="w", fontsize=fontsize2)
            else:
                ax.text(x_val, y_val, c, va='center', ha='center', color="k", fontsize=fontsize2)

    # convert axis vaues to to string labels

    y = y[::-1]
    #x = x[::-1]
    x=[str(i) for i in x]
    y=[str(np.log10(i)) for i in y]


    ax.set_xticklabels(['']+x, fontsize=fontsize2)
    ax.set_yticklabels(['']+y, fontsize=fontsize2)

    #ax.set_xlabel(r"$\\mathrm{\\ neurons}$",fontsize=fontsize)
    #ax.set_ylabel(r"$\\mathrm{epochs\\ }$",fontsize=fontsize)

    ax.set_xlabel(r"$\mathrm{\gamma} $",fontsize=fontsize)
    ax.set_ylabel(r"$\mathrm{\log_{10}\lambda} $",fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(figurename)

    plt.show()


#figurename = "grid_neurons_epochs.pdf"
#figurename = "grid_search_learningrate_datasz.pdf"
#figurename = "grid_search_regularization_lambda.pdf"
plot_data(x, y, z, figurename)

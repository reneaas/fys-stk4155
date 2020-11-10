import numpy as np
import matplotlib.pyplot as plt
import os

path ="./results/LogisticRegression/"

path_to_plot = "./results/LogisticRegression/Plots/"
if not os.path.exists(path_to_plot):
    os.makedirs(path_to_plot)

plotname = path_to_plot + "LogisticRegression_accuracy_analasys_broad.pdf"

gammas = np.load(path + "LogReg_Gamma_broad.npy")
etas = np.load(path + "LogReg_Eta_broad.npy")
accu = np.load(path + "LogReg_Accuracy_broad.npy")
etas = np.log10(etas)

accuracy = np.zeros([len(etas),len(gammas)])
accuracy.flat[:] = accu

def plot_heatmap_accuracy(x,y,accuracy, xlabel, ylabel, plotname, maxlabel):
    nogrid_x = np.copy(x)
    nogrid_y = np.copy(y)
    x, y = np.meshgrid(x, y)
    accuracy = accuracy*100

    idx_X, idx_Y = np.where(accuracy == np.max(accuracy))

    print("LogReg; max accuracy ", np.max(accuracy))
    print("eta= " ,nogrid_x[idx_X])
    print("gamma= " ,nogrid_y[idx_Y])
    print(nogrid_x)

    plot_name = plotname
    font_size = 12
    tick_size = 12
    plt.contourf(x, y, accuracy.T, cmap = "inferno", levels=40)
    for i in range(len(idx_X)):
        plt.plot(nogrid_x[idx_X[i]], nogrid_y[idx_Y[i]], "k+")
        plt.text(nogrid_x[idx_X[i]]-1, nogrid_y[idx_Y[i]]+0.05, maxlabel, color = "k", size=font_size)
    plt.xlabel(xlabel, size=font_size)
    plt.ylabel(ylabel, size=font_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    cb = plt.colorbar()
    cb.set_label(label="Accuracy (%)", size=font_size)
    cb.ax.tick_params(labelsize=font_size)
    plt.savefig(plot_name)
    plt.close()

plot_heatmap_accuracy(etas,gammas,accuracy,r"$\log_{10}(\eta)$", r"$\gamma$", plotname, "max accuracy" + r"$(\eta, \gamma)$" )

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

#plot_data(gammas, etas, accuracy)


plotname = path_to_plot + "LogisticRegression_accuracy_analasys_lambda.pdf"

Lambda = np.load(path + "LogReg_lambda.npy")
Lambda = np.log10(Lambda)
accu_lam = np.load(path + "LogReg_Accuracy_for_Lambda.npy")
#print(Lambda)
#print(accu_lam)

idx_L = np.where(accu_lam == np.max(accu_lam))
idx_L = idx_L[0]

plt.plot(Lambda, accu_lam)
plt.plot(Lambda[idx_L], accu_lam[idx_L], "m*")
plt.text(Lambda[idx_L]-0.5, accu_lam[idx_L], "Maximum accuracy for $\lambda$", color = "k", size=12)
plt.xlabel("$\lambda$")
plt.ylabel("Accuracy")
#plt.show()

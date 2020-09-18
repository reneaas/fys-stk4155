import numpy as np
import matplotlib.pyplot as plt
from ols import OLS


def plot_OLS_MSE_R2(filename,n,sigma):

    MSE_train = []
    MSE_test = []
    R2_train = []
    R2_test = []
    degrees = [1,2,3,4,5,7,8,9,10]
    solver = OLS()

    for deg in degrees:
        solver.read_data(filename,deg)
        solver.split_data()
        solver.train()
        solver.predict()
        MTrain, MTest = solver.extract_MSE()
        RTrain, RTest = solver.extract_R2()
        MSE_train.append(MTrain)
        MSE_test.append(MTest)
        R2_train.append(RTrain)
        R2_test.append(RTest)

    MSE_train = np.array(MSE_train)
    MSE_test = np.array(MSE_test)
    degrees = np.array(degrees)
    R2_train = np.array(R2_train)
    R2_test = np.array(R2_test)

    path = "./results/OLS/plots/"

    #Siden du separerer alle med "_" kan du bruke "_".join(liste_med_strings) for å sy de sammen til en string der alle liste elementene separeres med "_"
    plot_name = path + "_".join(["MSE", "N", str(n), "sigma", str(sigma)]) + ".pdf"
    #plot_name = path + "MSE_N_" + str(n) + "_sigma_" + str(sigma) + ".pdf"

    plt.plot(degrees, MSE_train, label = "Training data")
    plt.plot(degrees, MSE_test, label = "Test data")
    plt.title("MSE")
    plt.legend()
    plt.ylabel("Mean Square Error")
    plt.xlabel("Model Complexity")
    plt.savefig(plot_name)
    plt.figure()

    #Samme her, mer oversiktlig å bruke "_".join(liste_med_strings) for å sy de sammen med "_" som separering
    plot_name = path + "_".join(["R2", "N", str(n), "sigma", str(sigma)]) + ".pdf"
    #plot_name = path + "R2_N_" + str(n) + "_sigma_" + str(sigma) + ".pdf"

    plt.plot(degrees, R2_train, label = "Training data")
    plt.plot(degrees, R2_test, label = "Test data")
    plt.title("R2")
    plt.legend()
    plt.ylabel("R2 - Score")
    plt.xlabel("Model Complexity")
    plt.savefig(plot_name)

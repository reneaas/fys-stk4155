#from regression import Regression
from ols import OLS
from ridge import Ridge
from lasso import Lasso
#from plot import *
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

n = int(sys.argv[1]) #Number of datapoints
sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(n), "sigma", str(sigma)]) + ".txt"


regression_type = sys.argv[3]

filename = "./datasets/frankefunction_dataset_N_10000_sigma_1.0.txt"

def plot_OLS_k_fold_validation():

    MSE_train = []
    MSE_test = []
    R2_train = []
    R2_test = []
    bias = []
    variance = []
    degrees = [i for i in range(21)]
    k = 10
    solver = OLS()
    solver.read_data(filename)
    for deg in degrees:
        solver.create_design_matrix(deg)
        solver.split_data()
        solver.train(solver.X_train, solver.y_train)
        #solver.k_fold_cross_validation(k)
        RTrain, MTrain = solver.predict(solver.X_train, solver.y_train)
        RTest, MTest = solver.predict(solver.X_test, solver.y_test)
        Bias, Variance = solver.compute_bias_variance(solver.X_test, solver.y_test)
        MSE_train.append(MTrain)
        MSE_test.append(MTest)
        R2_train.append(RTrain)
        R2_test.append(RTest)
        bias.append(Bias)
        variance.append(Variance)


    MSE_train = np.array(MSE_train)
    MSE_test = np.array(MSE_test)
    degrees = np.array(degrees)
    R2_train = np.array(R2_train)
    R2_test = np.array(R2_test)
    bias = np.array(bias)
    variance = np.array(variance)


    """
    path = "./results/OLS/plots/kfold/"

    if not os.path.exists(path):
        os.makedirs(path)

    plot_name = path + "_".join(["k_fold",str(k),"MSE", "N", str(n), "sigma", str(sigma)]) + ".pdf"

    plt.plot(degrees, MSE_train, label = "Training data")
    plt.plot(degrees, MSE_test, label = "Test data")
    plt.title("%i-fold cross validation - MSE" %k)
    plt.legend()
    plt.ylabel("Mean Square Error")
    plt.xlabel("Model Complexity")
    plt.savefig(plot_name)
    plt.figure()


    plot_name = path + "_".join(["k_fold",str(k),"R2", "N", str(n), "sigma", str(sigma)]) + ".pdf"


    plt.plot(degrees, R2_train, label = "Training data")
    plt.plot(degrees, R2_test, label = "Test data")
    plt.title("%i-fold cross validation - R2" % k)
    plt.legend()
    plt.ylabel("R2 - Score")
    plt.xlabel("Model Complexity")
    plt.savefig(plot_name)
    """
    plt.plot(degrees, R2_train, label="$R^2$ (training)")
    plt.plot(degrees, R2_test, label = "$R^2$ (test)")
    plt.figure()
    plt.plot(degrees, MSE_test, label="$E_{out}$")
    plt.plot(degrees, MSE_train, label="$E_{in}$")
    plt.plot(degrees, bias, "--", label="bias")
    plt.plot(degrees, variance, "o-", label="variance")
    plt.legend()
    plt.show()

if regression_type == "OLS":
    plot_OLS_k_fold_validation()

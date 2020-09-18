from regression import Regression
from ols import OLS
from ridge import Ridge
from lasso import Lasso
from plot import *
import numpy as np
import sys
import os

n = int(sys.argv[1])
sigma = float(sys.argv[2])
path_to_datasets = "./datasets/"
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(n), "sigma", str(sigma)]) + ".txt"

regression_type = sys.argv[3]


def plot_OLS_bootstrap():

    solver = OLS()
    solver.read_data(filename)

    degrees = [i for i in range(1,14)]
    p = len(degrees)
    b = 1000
    MSE_train = np.zeros(p)
    MSE_test = np.zeros(p)
    R2_train = np.zeros(p)
    R2_test = np.zeros(p)

    for i in range(p):
        solver.create_design_matrix(degrees[i])
        solver.split_data()
        solver.bootstrap(b)
        R2_train[i], MSE_train[i] = solver.predict_train()
        R2_test[i], MSE_test[i] = solver.predict_test()

    lowest_MSE_val_idx = np.argmin(MSE_test)
    print("Polynomial degree which provides lowest MSE score on test data: %i" % (lowest_MSE_val_idx+1))

    path = "./results/OLS/plots/bootstrap/"
    if not os.path.exists(path):
        os.makedirs(path)

    plot_name = path + "_".join(["Bootstrap",str(b),"MSE", "N", str(n), "sigma", str(sigma)]) + ".pdf"

    plt.plot(degrees, MSE_train, label = "Training data")
    plt.plot(degrees, MSE_test, label = "Test data")
    plt.title("Bootstrap - MSE")
    plt.legend()
    plt.ylabel("Mean Square Error")
    plt.xlabel("Model Complexity")
    plt.savefig(plot_name)
    plt.figure()


    plot_name = path + "_".join(["Bootstrap",str(b),"R2", "N", str(n), "sigma", str(sigma)]) + ".pdf"


    plt.plot(degrees, R2_train, label = "Training data")
    plt.plot(degrees, R2_test, label = "Test data")
    plt.title("Bootstrap - R2")
    plt.legend()
    plt.ylabel("R2 - Score")
    plt.xlabel("Model Complexity")
    plt.savefig(plot_name)

if regression_type =="OLS":
    plot_OLS_bootstrap()

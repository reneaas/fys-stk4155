from regression import Regression
from ols import OLS
from ridge import Ridge
from lasso import Lasso
from plot import *
import numpy as np
import sys
import os

n = int(sys.argv[1]) #Number of datapoints
sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(n), "sigma", str(sigma)]) + ".txt"


regression_type = sys.argv[3]


def plot_OLS_k_fold_validation():

    MSE_train = []
    MSE_test = []
    R2_train = []
    R2_test = []
    degrees = [1,2,3,4,5,7,8,9,10,11,12,13,14,15]
    k = 1000
    solver = OLS()

    for deg in degrees:
        solver.read_data(filename,deg)
        solver.split_data()
        solver.k_fold_cross_validation(k)
        RTrain, MTrain = solver.predict(solver.X_train, solver.y_train)
        RTest, MTest = solver.predict(solver.X_test, solver.y_test)
        MSE_train.append(MTrain)
        MSE_test.append(MTest)
        R2_train.append(RTrain)
        R2_test.append(RTest)

    MSE_train = np.array(MSE_train)
    MSE_test = np.array(MSE_test)
    degrees = np.array(degrees)
    R2_train = np.array(R2_train)
    R2_test = np.array(R2_test)

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

if regression_type == "OLS":
    plot_OLS_k_fold_validation()

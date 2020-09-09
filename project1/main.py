from regression import Regression
from ols import OLS
from ridge import Ridge
from plot import *
import numpy as np
import sys
import os

n = int(sys.argv[1]) #Number of datapoints
sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(n), "sigma", str(sigma)]) + ".txt"

#plot_OLS_MSE_R2(filename, n, sigma) #Plots the MSE and R2-score of the OLS regression


polynomial_degree = 2 #Maximum degree of polynomial
solver = OLS() #Initiate solver
#solver = Ridge(Lambda = 0.001)
solver.read_data(filename, polynomial_degree) #Read data from file
solver.split_data()
#solver.bootstrap(100)
solver.k_fold_cross_validation(3)

#Ridge Regularization path
"""
python_path = "./compute_regularization_path.py"
os.system(" ".join(["python3", python_path, str(n), str(sigma)]))
"""

solver = Ridge(Lambda = 0.001)
solver.read_data(filename, polynomial_degree) #Read data from file
solver.split_data()
filename_plots = ["regularization_path_R2.pdf", "regularization_path_MSE.pdf"]
path_plots = "./results/plots"
solver.plot_regularization_path(filename_plots, path_plots)
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

filenames = "_".join(filename_plots)
os.system(" ".join(["mv", path_plots, filenames]))

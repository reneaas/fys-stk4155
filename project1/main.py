from regression import *
from ols import *
from plot import *
import numpy as np
import sys

n = int(sys.argv[1]) #Number of datapoints
sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
filename = path_to_datasets + "frankefunction_dataset_N_" + str(n) + "_sigma_" + str(sigma) + ".txt"

plot_OLS_MSE_R2(filename, n, sigma) #Plots the MSE and R2-score of the OLS regression

polynomial_degree = 2 #Maximum degree of polynomial
solver = OLS() #Initiate solver
solver.read_data(filename, polynomial_degree) #Read data from file
solver.split_data()
solver.bootstrap(100)

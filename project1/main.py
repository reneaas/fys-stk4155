from regression import *
from ols import *
import numpy as np
import sys

n = int(sys.argv[1]) #Number of datapoints
sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
filename = path_to_datasets + "frankefunction_dataset_N_" + str(n) + "_sigma_" + str(sigma) + ".txt"

polynomial_degree = 2 #Maximum degree of polynomial
solver = OLS() #Initiate solver
solver.read_data(filename, polynomial_degree) #Read data from file
solver.split_data() #Split the data into training and test set
solver.train()   #Perform ordinary least squares
solver.predict() #Predict and compute R2 scores.
solver.bootstrap()
#solver.confidence_intervals(0.1)
#solver.Bootstrap(3,5)

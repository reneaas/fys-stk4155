from Regression import *
import numpy as np
import sys

n = int(sys.argv[1]) #Number of datapoints
sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
filename = path_to_datasets + "frankefunction_dataset_N_" + str(n) + "_sigma_" + str(sigma) + ".txt"

polynomial_degree = 5 #Maximum degree of polynomial
solver = Regression() #Initiate solver
solver.ReadData(filename, polynomial_degree) #Read data from file
solver.SplitData() #Split the data into training and test set
solver.OLS()   #Perform ordinary least squares
#solver.Predict() #Predict and compute R2 scores.
solver.kfold_CrossValidation(5)

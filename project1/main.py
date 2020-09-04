from Regression import *
import numpy as np
import sys

<<<<<<< HEAD
#path_to_datasets = "/Users/kasparagasvaer/Documents/FYS-STK4155/fys-stk4155/Project1/datasets/"
n = int(sys.argv[1])
path_to_datasets = "./datasets/"

filename = path_to_datasets + "frankefunction_dataset_N_" + str(n) + "_sigma_0.1.txt"

deg = 5
#Ender opp med å være 20 attributes, can't for the life of me generalisere
#(hint: kombinatorikk, skal ikke ha med 0,0, men alle andre opp til 5 grad)
#Vil egt sende inn typ 5, også fikse kombinatorikk magic i ReadData.


#vil vi vite punkter på forhånd eller finne dette ut i ReadData??

solver = Regression(n)
solver.ReadData(filename, 5)
solver.SplitData()
solver.OLS()
solver.Predict()
solver.MSE()
#f_train, f_test, f_tilde, f_predict = solver.LinearRegression()

#Training
#R2_train = solver.R2(f_train, f_tilde)
#MSE_train = solver.MSE(f_train, f_tilde)

#Testing
#R2_test = solver.R2(f_test, f_predict)
#MSE_test = solver.MSE(f_test, f_predict)
=======
n = int(sys.argv[1]) #Number of datapoints
sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
filename = path_to_datasets + "frankefunction_dataset_N_" + str(n) + "_sigma_" + str(sigma) + ".txt"

polynomial_degree = 5 #Maximum degree of polynomial
solver = Regression() #Initiate solver
solver.ReadData(filename, polynomial_degree) #Read data from file
solver.SplitData() #Split the data into training and test set
solver.OLS()   #Perform ordinary least squares
solver.Predict() #Predict and compute R2 scores.
>>>>>>> e5d0b976eb20fecbe1da820e721eebfa278991a1

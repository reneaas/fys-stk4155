from Regression import *
import numpy as np

path_to_datasets = "/Users/kasparagasvaer/Documents/FYS-STK4155/fys-stk4155/Project1/datasets/"

filename = path_to_datasets + "frankefunction_dataset_N_1000_sigma_0.1.txt"

deg = 5
#Ender opp med å være 20 attributes, can't for the life of me generalisere
#(hint: kombinatorikk, skal ikke ha med 0,0, men alle andre opp til 5 grad)
#Vil egt sende inn typ 5, også fikse kombinatorikk magic i ReadData.

n = 1000
#vil vi vite punkter på forhånd eller finne dette ut i ReadData??

solver = Regression(n)
solver.ReadData(filename, deg)
f_train, f_test, f_tilde, f_predict = solver.LinearRegression()

#Training
#R2_train = solver.R2(f_train, f_tilde)
#MSE_train = solver.MSE(f_train, f_tilde)

#Testing
#R2_test = solver.R2(f_test, f_predict)
#MSE_test = solver.MSE(f_test, f_predict)

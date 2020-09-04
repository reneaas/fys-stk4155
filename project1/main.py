from regression import *
from ols import *
import numpy as np
import sys

n = int(sys.argv[1]) #Number of datapoints
sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
filename = path_to_datasets + "frankefunction_dataset_N_" + str(n) + "_sigma_" + str(sigma) + ".txt"



"""
polynomial_degree = 2 #Maximum degree of polynomial
solver = OLS() #Initiate solver
solver.read_data(filename, polynomial_degree) #Read data from file
solver.split_data()
solver.bootstrap(100)
#solver.confidence_intervals(0.1)
#solver.Bootstrap(3,5)
"""


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

plt.plot(degrees, MSE_train, label = "Training data")
plt.plot(degrees, MSE_test, label = "Test data")
plt.title("MSE")
plt.legend()
plt.ylabel("Mean Square Error")
plt.xlabel("Model Complexity")
plt.show()

plt.plot(degrees, R2_train, label = "Training data")
plt.plot(degrees, R2_test, label = "Test data")
plt.title("R2")
plt.legend()
plt.ylabel("R2 - Score")
plt.xlabel("Model Complexity")
plt.show()

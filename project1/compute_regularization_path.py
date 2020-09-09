from ridge import Ridge
import matplotlib.pyplot as plt
import sys
import numpy as np
plt.rc("text", usetex=True)

n = int(sys.argv[1]) #Number of datapoints
sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(n), "sigma", str(sigma)]) + ".txt"

#Testing Ridge regression module
polynomial_degree = 2 #Maximum degree of polynomial
Lambda = [1/10**i for i in range(7)]
print(Lambda)
R2_scores_train = []
MSE_scores_train = []
R2_scores_test = []
MSE_scores_test = []

for l in Lambda:
    print("lambda = ", l)
    solver = Ridge(l) #Initiate solver
    solver.read_data(filename, polynomial_degree) #Read data from file
    solver.split_data()
    #solver.bootstrap(100)
    solver.train(solver.X_train, solver.y_train)
    R2_train, MSE_train = solver.predict(solver.X_train, solver.y_train)
    R2_test, MSE_test = solver.predict(solver.X_test, solver.y_test)
    R2_scores_train.append(R2_train)
    MSE_scores_train.append(MSE_train)
    R2_scores_test.append(R2_test)
    MSE_scores_test.append(MSE_test)

plt.plot(np.log10(Lambda), R2_scores_train, label = "Training")
plt.plot(np.log10(Lambda), R2_scores_test, label = "Test")
plt.xlabel(r"$\log_{10}( \lambda) $")
plt.ylabel(r"$R^2$")
plt.show()

plt.plot(np.log10(Lambda), MSE_scores_train, label = "Training")
plt.plot(np.log10(Lambda), MSE_scores_test, label = "Test")
plt.xlabel(r"$\log_{10}( \lambda) $")
plt.ylabel("MSE")
plt.show()

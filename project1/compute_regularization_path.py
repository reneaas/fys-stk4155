from ridge import Ridge
from lasso import Lasso
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
plt.rc("text", usetex=True)

#n = int(sys.argv[1]) #Number of datapoints
#sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
#filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(n), "sigma", str(sigma)]) + ".txt"
<<<<<<< HEAD
filename = "terrain_data.txt"

filename_plots = ["regularization_path_R2_p5.pdf", "regularization_path_MSE_p5.pdf"]
path_plots = "./results/plots/terrain"


#Testing Ridge regression module
polynomial_degree = 5 #Maximum degree of polynomial
Lambda = [1/10**i for i in range(-2,2+1)]
=======
filename = path_to_datasets + "terrain_data.txt"
filename_plots = ["regularization_path_R2.pdf", "regularization_path_MSE.pdf"]
path_plots = "./results/plots"


#Testing Ridge regression module
polynomial_degree = 2 #Maximum degree of polynomial
Lambda = [1/10**i for i in range(-10,10)]
>>>>>>> 586fb7d29f5413b4ce72e8f4f5d6ba68d5f9b216
R2_scores_train = [[],[]]
MSE_scores_train = [[],[]]
R2_scores_test = [[],[]]
MSE_scores_test = [[],[]]

#Compute regularization path for Ridge
print("Ridge")
solver = Ridge(0.0001)
solver.read_data(filename, polynomial_degree) #Read data from file
solver.split_data()
for l in Lambda:
    print("lambda = ", l)
    solver.Lambda = l
    solver.train(solver.X_train, solver.y_train)
    R2_train, MSE_train = solver.predict(solver.X_train, solver.y_train)
    R2_test, MSE_test = solver.predict(solver.X_test, solver.y_test)
    R2_scores_train[0].append(R2_train)
    MSE_scores_train[0].append(MSE_train)
    R2_scores_test[0].append(R2_test)
    MSE_scores_test[0].append(MSE_test)

print("Lasso")
solver = Lasso(0.0001)
solver.read_data(filename, polynomial_degree) #Read data from file
solver.split_data()
for l in Lambda:
    print("lambda = ", l)
    solver.Lambda = l
    solver.train(solver.X_train, solver.y_train)
    R2_train, MSE_train = solver.predict(solver.X_train, solver.y_train)
    R2_test, MSE_test = solver.predict(solver.X_test, solver.y_test)
    R2_scores_train[1].append(R2_train)
    MSE_scores_train[1].append(MSE_train)
    R2_scores_test[1].append(R2_test)
    MSE_scores_test[1].append(MSE_test)

plt.plot(np.log10(Lambda), R2_scores_train[0], label = "Training (Ridge)")
plt.plot(np.log10(Lambda), R2_scores_test[0], label = "Test (Ridge)")
plt.plot(np.log10(Lambda), R2_scores_train[1], label = "Training (Lasso)")
plt.plot(np.log10(Lambda), R2_scores_test[1], label = "Test (Lasso)")
plt.xlabel(r"$\log_{10}( \lambda) $", fontsize=14)
plt.ylabel(r"$R^2$", fontsize=14)
plt.legend(fontsize=14)
plt.savefig(filename_plots[0])
plt.figure()

plt.plot(np.log10(Lambda), MSE_scores_train[0], label = "Training (Ridge)")
plt.plot(np.log10(Lambda), MSE_scores_test[0], label = "Test (Ridge)")
plt.plot(np.log10(Lambda), MSE_scores_train[1], label = "Training (Lasso)")
plt.plot(np.log10(Lambda), MSE_scores_test[1], label = "Test (Lasso)")
plt.xlabel(r"$\log_{10}( \lambda) $", fontsize=14)
plt.ylabel("MSE", fontsize=14)
plt.legend(fontsize=14)
plt.savefig(filename_plots[1])
plt.show()

if not os.path.exists(path_plots):
    os.makedirs(path_plots)

filenames = " ".join(filename_plots)
os.system(" ".join(["mv", filenames, path_plots]))

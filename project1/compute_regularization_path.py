from ridge import Ridge
from lasso import Lasso
from ols import OLS
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
plt.rc("text", usetex=True)

#n = int(sys.argv[1]) #Number of datapoints
#sigma = float(sys.argv[2]) #Standard deviation of noise from data
path_to_datasets = "./datasets/" #relative path into subdirectory for datasets.
#filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(n), "sigma", str(sigma)]) + ".txt"

filename = "terrain_data_small.txt"

filename = "terrain_data.txt"

filename_plots = ["regularization_path_R2_p5.pdf", "regularization_path_MSE_p5.pdf"]
path_plots = "./results/plots/terrain"



#filename = path_to_datasets + "terrain_data.txt"
#filename = "./datasets/frankefunction_dataset_N_1000_sigma_1.0.txt"
filename = "terrain_data_small.txt"
filename_plots = ["regularization_path_R2.pdf", "regularization_path_MSE.pdf"]
path_plots = "./results/plots"


#Testing Ridge regression module
polynomial_degree = 2 #Maximum degree of polynomial
Lambda = [1/10**i for i in range(2,10)]

R2_scores_train = [[],[]]
MSE_scores_train = [[],[]]
R2_scores_test = [[],[]]
MSE_scores_test = [[],[]]

#Compute regularization path for Ridge
print("Ridge")
solver = Ridge(0.0001)
solver.read_data(filename) #Read data from file
solver.create_design_matrix(polynomial_degree)
solver.split_data()
for l in Lambda:
    print("lambda = ", l)
    solver.Lambda = l
    solver.train()
    R2_train, MSE_train = solver.predict_train()
    R2_test, MSE_test = solver.predict_test()
    R2_scores_train[0].append(R2_train)
    MSE_scores_train[0].append(MSE_train)
    R2_scores_test[0].append(R2_test)
    MSE_scores_test[0].append(MSE_test)

print("Lasso")
solver = Lasso(0.0001)
solver.read_data(filename) #Read data from file
solver.create_design_matrix(polynomial_degree)
solver.split_data()
for l in Lambda:
    print("lambda = ", l)
    solver.Lambda = l
    solver.train()
    R2_train, MSE_train = solver.predict_train()
    R2_test, MSE_test = solver.predict_test()
    R2_scores_train[1].append(R2_train)
    MSE_scores_train[1].append(MSE_train)
    R2_scores_test[1].append(R2_test)
    MSE_scores_test[1].append(MSE_test)

solver = OLS()
solver.read_data(filename) #Read data from file
solver.create_design_matrix(polynomial_degree)
solver.split_data()
solver.train()
R2_train, MSE_train = solver.predict_train()
R2_test, MSE_test = solver.predict_test()

print("Training R2 = ", R2_train)
print("Test R2 = ", R2_test)
print("Training MSE = ", MSE_train)
print("Test MSE = ", MSE_test)

plt.plot(np.log10(Lambda), R2_scores_train[0], "--", label = "Training (Ridge)")
plt.plot(np.log10(Lambda), R2_scores_test[0], label = "Test (Ridge)")
plt.plot(np.log10(Lambda), R2_scores_train[1], "--",label = "Training (Lasso)")
plt.plot(np.log10(Lambda), R2_scores_test[1], label = "Test (Lasso)")
plt.axhline(R2_train, color = "k" ,  label = "Training (OLS)", linestyle = "--")
plt.axhline(R2_test, color="r" , label = "Test (OLS)", linestyle = "-")
plt.xlabel(r"$\log_{10}( \lambda) $", fontsize=14)
plt.ylabel(r"$R^2$", fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(fontsize=14)
plt.savefig(filename_plots[0])
plt.figure()

plt.plot(np.log10(Lambda), MSE_scores_train[0], "--", label = "Training (Ridge)")
plt.plot(np.log10(Lambda), MSE_scores_test[0], label = "Test (Ridge)")
plt.plot(np.log10(Lambda), MSE_scores_train[1], "--", label = "Training (Lasso)")
plt.plot(np.log10(Lambda), MSE_scores_test[1], label = "Test (Lasso)")
plt.axhline(MSE_train, color = "k" ,  label = "Training (OLS)", linestyle = "--")
plt.axhline(MSE_test, color="r" , label = "Test (OLS)", linestyle = "-")
plt.xlabel(r"$\log_{10}( \lambda) $", fontsize=14)
plt.ylabel("MSE", fontsize=14)
plt.legend(fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.savefig(filename_plots[1])
plt.show()

if not os.path.exists(path_plots):
    os.makedirs(path_plots)

filenames = " ".join(filename_plots)
os.system(" ".join(["mv", filenames, path_plots]))

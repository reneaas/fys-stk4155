from ridge import Ridge
from lasso import Lasso
from ols import OLS
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
plt.rc("text", usetex=True)


filename = "./datasets/TerrainFiles/terrain_data.txt"
#filename = "./datasets/frankefunction_dataset_N_1000_sigma_0.1.txt"

method = sys.argv[1]
Polynomial_degrees = [i for i in range(24)]
P = len(Polynomial_degrees)
k = 10

if method == "Ridge":

    Lambdas = [1/10**i for i in range(-2,7)]

    L = len(Lambdas)
    MSE_Test_Ridge = np.zeros([L,P])
    R2_Test_Ridge = np.zeros([L,P])

    solver = Ridge()
    solver.read_data(filename)
    for i in range(P):
        print("p = ", i)
        solver.create_design_matrix(Polynomial_degrees[i])
        solver.split_data()
        for j in range(L):
            solver.Lambda = Lambdas[j]
            R2, MSE = solver.k_fold_cross_validation(k)
            MSE_Test_Ridge[j,i] = MSE
            R2_Test_Ridge[j,i] = R2

    np.save("MSE_Test_Ridge_Terrain.npy", MSE_Test_Ridge)
    np.save("R2_Test_Ridge_Terrain.npy", R2_Test_Ridge)


if method == "Lasso":

    Lambdas = [1/10**i for i in range(-1,6)]
    L = len(Lambdas)
    MSE_Test_Lasso = np.zeros([L,P])
    R2_Test_Lasso = np.zeros([L,P])

    solver = Lasso()
    solver.read_data(filename)
    for i in range(P):
        print("p = ", i)
        solver.create_design_matrix(Polynomial_degrees[i])
        solver.split_data()
        for j in range(L):
            solver.Lambda = Lambdas[j]
            solver.train()
            R2, MSE = solver.predict_test()
            #R2, MSE = solver.k_fold_cross_validation(k)
            MSE_Test_Lasso[j,i] = MSE
            R2_Test_Lasso[j,i] = R2

    np.save("MSE_Test_Lasso_Terrain.npy", MSE_Test_Lasso)
    np.save("R2_Test_Lasso_Terrain.npy", R2_Test_Lasso)

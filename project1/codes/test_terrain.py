from regression import OLS, Ridge, Lasso
import numpy as np

dataset1 = "../datasets/TerrainFiles/terrain_data_1.txt"
dataset2 = "../datasets/TerrainFiles/terrain_data_2.txt"

degs = [i for i in range(26)]
lambdas = [1/10**i for i in range(-5, 6)]

P = len(degs)
L = len(lambdas)

MSE = np.zeros((L,P))
R2 = np.zeros((L,P))


k = 10
solver = Ridge()
solver.read_data(dataset1)
for i in range(P):
    print("d = ", i)
    solver.create_design_matrix(degs[i])
    solver.split_data()
    for j in range(L):
        solver.Lambda = lambdas[j]
        r2, mse = solver.k_fold_cross_validation(k)
        MSE[j, i] = mse
        R2[j, i] = r2


B = 1000
l, d = np.where(MSE == np.min(MSE))
l = l[0]; d = d[0]
solver = Ridge(Lambda = l)
solver.read_data(dataset2)
solver.create_design_matrix(d)
solver.split_data()
r2, mse, bias, variance = solver.bootstrap(B)

print("MSE = ", mse)
print("R2 = ", r2)

from regression import OLS, Ridge, Lasso
import numpy as np
import matplotlib.pyplot as plt
plt.rc("text", usetex=True)

dataset1 = "../datasets/TerrainFiles/terrain_data_1.txt"
dataset2 = "../datasets/TerrainFiles/terrain_data_2.txt"
"""
degs = [i for i in range(26)]
lambdas = [1/10**i for i in range(-5, 6)]

P = len(degs)
L = len(lambdas)

MSE = np.zeros((P, L))
R2 = np.zeros((P, L))


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
        MSE[i, j] = mse
        R2[i, j] = r2

d, l = np.where(MSE == np.min(MSE))
l = l[0]; d = d[0]
print("d = ", d)
print("l = ", l)
print(degs[d])
print("minimum MSE = ", np.min(MSE))
lamb = lambdas[l]
deg = degs[d]

lambdas = np.log10(lambdas)
Degs, Lambdas = np.meshgrid(degs, lambdas)

font_size = 14
tick_size = 14
plt.contourf(Degs, Lambdas, MSE.T, cmap = "inferno", levels=40)
plt.plot(degs[d], lambdas[l], "w+")
plt.text(degs[d] - 6, lambdas[l] + 0.3, "min MSE" + r"$(d, \lambda)$", color = "w", size=14)
plt.xlabel(r"$d$",size=font_size)
plt.ylabel(r"$\log_{10}(\lambda)$", size=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
cb = plt.colorbar()
cb.set_label(label="MSE", size=14)
cb.ax.tick_params(labelsize=14)
plt.show()
"""


k = 10
B = 1000
lamb = 1e-4
deg = 19
solver1 = Lasso(Lambda = lamb)
solver1.read_data(dataset1)
solver1.create_design_matrix(deg)
solver1.split_data()

solver2 = Lasso(Lambda = lamb)
solver2.read_data(dataset2)
solver2.create_design_matrix(deg)
solver2.split_data()

solver1.X_test = solver2.X_test
solver1.f_test = solver2.f_test


r2, mse, bias, variance = solver1.bootstrap(B)

print("MSE = ", mse)
print("R2 = ", r2)

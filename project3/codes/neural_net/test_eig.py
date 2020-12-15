from neural_eigensolver import NeuralEigenSolver
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

seed = 100
tf.random.set_seed(100)
np.random.seed(100)

mat_sz = 6
A = np.load("6_by_6matrix.npy")
mat_sz = np.shape(A)[0]
true_eigvals, true_eigvecs = np.linalg.eig(A)

#Initialize the model
input_sz = 1
layers = [500, mat_sz]
eig_type = "max"
my_solver = NeuralEigenSolver(layers = layers, input_sz = input_sz, matrix = A, eig_type = eig_type)

#Fit the model
Nt = 10
t_max = 1e3
x = np.random.normal(0, 1, size=mat_sz)
idx = np.where(true_eigvals == np.max(true_eigvals))
max_eigvec = true_eigvecs.T[idx]
x = np.copy(max_eigvec)

t = np.linspace(0, t_max, Nt)
epochs = 100
epoch_arr, eigvals, eigvecs = my_solver.fit(x = x, t = t, epochs = epochs)

#Plot eigenvalue estimate as function of epochs
fontsize = 16
plt.plot(epoch_arr, eigvals, label= "eigenvalue estimate", color="r")
plt.hlines(y = true_eigvals, xmin = 0, xmax = epochs, linestyles="dashed")
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel("epochs", size=fontsize)
plt.ylabel("eigenvalue", size=fontsize)
plt.legend(fontsize=fontsize)
plt.show()

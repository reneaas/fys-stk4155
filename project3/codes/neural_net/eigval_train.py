from neural_eigensolver import NeuralEigenSolver
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

#Create symmetric matrix
# mat_sz = 3
# A = np.array([[3, 0, 4], [0, 2, 0], [4, 0, 3]])
mat_sz = 6
A = np.load("matrix.npy")
mat_sz = np.shape(A)[0]
print(A)


#Initialize the model
input_sz = 1
layers = [10000, 100, 10000, 100, mat_sz]
eig_type = "max"
my_solver = NeuralEigenSolver(layers = layers, input_sz = input_sz, matrix = A, eig_type = eig_type)
# np.save("matrix.npy", A)

print(my_solver.summary())


#Fit the model
Nt = 250
t_max = 1000
x = np.random.normal(0, 1, size=mat_sz)

t = np.linspace(0, t_max, Nt)
epochs = 2
epoch_arr, eigvals, eigvecs = my_solver.fit(x = x, t = t, epochs = epochs)

my_solver.save("saved_model/eigenvalue_model")

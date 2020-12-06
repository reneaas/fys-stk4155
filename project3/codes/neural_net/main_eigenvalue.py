from neural_eigensolver import NeuralEigenSolver
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

#Set up solver
mat_sz = 3
A = np.array([[3, 0, 4], [0, 2, 0], [4, 0, 3]])
# Q = np.random.normal(0, 1, size=(mat_sz, mat_sz))
# A = 0.5*(Q.T + Q)

#Initialize the model
input_sz = 1
layers = [1, 100, 100, 3]
my_solver = NeuralEigenSolver(layers = layers, input_sz = input_sz, matrix = A)


#Fit the model
Nt = 100
t_max = 1000
x = np.random.normal(0, 1, size=mat_sz)
t = np.linspace(0, t_max, Nt)
epochs = 1000
my_solver.fit(x = x, t = t, epochs = epochs)

#Compute predictions. Computed eigenvalues converge as t --> inf.
T = np.linspace(0, 1.1*t_max, 1001)
eigenvalues = np.zeros_like(T)
eigenvectors = np.zeros([len(T), mat_sz])
for i in range(len(T)):
    t = tf.constant(np.array([T[i]]).reshape(-1,1), dtype=tf.float32)
    eigenvalue, eigenvector = my_solver.eig(x, t)
    eigenvalues[i] = eigenvalue
    for j in range(mat_sz):
        eigenvectors[i,j] = eigenvector.numpy()[j]

true_eigenvalues, true_eigenvectors = np.linalg.eig(A)
print("true eigenvalues = \n", true_eigenvalues)
print("True eigenvectors = \n", true_eigenvectors)
print("Estimated eigenvector =  \n", eigenvector.numpy())
print("Estimated eigenvalue = ", eigenvalues[-1]) #Largest value for t.


fontsize = 12
plt.plot(T, eigenvalues)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.show()


for j in range(mat_sz):
    plt.plot(T, eigenvectors[:,j], label=f"$x_{j}$")
plt.xlabel(r"$t$", size=fontsize)
plt.ylabel(r"$x_i(t)$", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.show()

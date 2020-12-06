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
epochs = 100
epoch_arr, eigvals, eigvecs = my_solver.fit(x = x, t = t, epochs = epochs)


#Plot eigenvalue estimate as function of epochs
fontsize = 12
plt.plot(epoch_arr, eigvals)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel("epochs", size=fontsize)
plt.ylabel("eigenvalue estimate", size=fontsize)
plt.show()

#Plot eigenvector estimate as function of epochs
for j in range(mat_sz):
    plt.plot(epoch_arr, eigvecs[:,j], label=f"$x_{j}$")
plt.xlabel("epochs", size=fontsize)
plt.ylabel(r"$x_i$", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.show()

#Compute predictions. Computed eigenvalues converge as t --> inf.
T = np.linspace(0, 1.1*t_max, 1001)
eigvals = np.zeros_like(T)
eigvecs = np.zeros([len(T), mat_sz])
for i in range(len(T)):
    t = tf.constant(np.array([T[i]]).reshape(-1,1), dtype=tf.float32)
    eigval, eigvec = my_solver.eig(x, t)
    eigvals[i] = eigval
    for j in range(mat_sz):
        eigvecs[i,j] = eigvec.numpy()[j]

true_eigvals, true_eigvecs = np.linalg.eig(A)
print("true eigenvalues = \n", true_eigvals)
print("True eigenvectors = \n", true_eigvecs)
print("Estimated eigenvector =  \n", eigvec.numpy())
print("Estimated eigenvalue = ", eigvals[-1]) #Largest value for t.


#Plot "eigenvalue prediction" as function of t for final model. Converges to true as t --> inf
plt.plot(T, eigvals)
plt.ylabel("eigenvalue estimate", size=fontsize)
plt.xlabel(r"$t$", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plt.legend(fontsize=fontsize)
plt.show()


#plot "eigenvector prediction" as function of t for final model. Converges to true as t --> inf
for j in range(mat_sz):
    plt.plot(T, eigvecs[:,j], label=f"$x_{j}$")
plt.xlabel(r"$t$", size=fontsize)
plt.ylabel(r"$x_i(t)$", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.show()

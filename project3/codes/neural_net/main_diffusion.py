import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neural_diffusion_solver import NeuralDiffusionSolver

seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

exact = lambda x, t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)

layers = [1000, 500, 1000, 1]
input_sz = 2
n = 100

x = np.random.uniform(0, 1, n)
t = np.random.uniform(0, 1, n)

epochs = 500

my_model = NeuralDiffusionSolver(layers, input_sz)
epoch_arr, loss = my_model.fit(x=x, t=t, epochs=epochs)

fontsize = 12
plt.plot(epoch_arr, loss)
plt.xlabel("epochs", size=fontsize)
plt.ylabel("loss", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()


# Define grid
num_points = 41
start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(1, dtype=tf.float32)
start_t = tf.constant(0, dtype=tf.float32)
stop_t = tf.constant(1, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start_t, stop_t, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])


f_predict = my_model.predict(x, t)
g = tf.reshape(exact(x, t), (num_points, num_points))
g_nn = tf.reshape(f_predict, (num_points, num_points))

G = g.numpy().ravel()
G_NN = g_nn.numpy().ravel()
res = np.sum((G-G_NN)**2)
tot = np.sum((G - np.mean(G))**2)
R2 = 1 - res/tot
print("R2 = ", R2)



plt.contourf(X, T, g, levels=41, cmap="inferno")
plt.colorbar()

fig = plt.figure()
plt.contourf(X, T, g_nn, levels=41, cmap="inferno")
plt.colorbar()
plt.show()

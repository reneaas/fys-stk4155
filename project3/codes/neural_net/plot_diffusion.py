import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neural_diffusion_solver import NeuralDiffusionSolver

seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

loaded_model = tf.keras.models.load_model('my_pde_model')

exact = lambda x, t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)

layers = [1000, 1000, 1000, 1000, 1]
input_sz = 2
n = 100

x = np.random.uniform(0, 1, n)
t = np.random.uniform(0, 1, n)

epochs = 500

fontsize = 14

# Define grid
num_points = 41
start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(1, dtype=tf.float32)
start_t = tf.constant(0, dtype=tf.float32)
stop_t = tf.constant(1, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])


f_predict = loaded_model.predict(x, t)
g = tf.reshape(exact(x, t), (num_points, num_points))
g_nn = tf.reshape(f_predict, (num_points, num_points))

G = g.numpy()
G_NN = g_nn.numpy()
G = G[1:,1:-1]
G_NN = G_NN[1:,1:-1]

rel_error = np.abs(G_NN - G)/G
rel_x = np.linspace(0,1,num_points)
rel_x = rel_x[1:-1]
rel_t = np.linspace(0,1,num_points)
rel_t = rel_t[1:]


RX,RT = np.meshgrid(rel_x,rel_t)

plt.contourf(X, T, g, levels=41, cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
cb = plt.colorbar()
cb.set_label(label=r"$u(x,t)$", size=fontsize)
plt.savefig("../results/neural_net/pde/pde_analytical_on_training_domain.pdf")

fig = plt.figure()
plt.contourf(X, T, g_nn, levels=41, cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
cb = plt.colorbar()
cb.set_label(label=r"$u(x,t)$", size=fontsize)
plt.savefig("../results/neural_net/pde/pde_network_on_training_domain.pdf")
plt.show()

fig2 = plt.figure()
plt.contourf(RX, RT, rel_error, levels=41, cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
cb = plt.colorbar()
cb.set_label(label="Relative Error", size=fontsize)
plt.savefig("../results/neural_net/pde/pde_rel_error_on_training_domain.pdf")
plt.show()

G = g.numpy().ravel()
G_NN = g_nn.numpy().ravel()
res = np.sum((G-G_NN)**2)
tot = np.sum((G - np.mean(G))**2)
R2 = 1 - res/tot
print("R2 on domain = ", R2)


start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(1, dtype=tf.float32)
start_t = tf.constant(1, dtype=tf.float32)
stop_t = tf.constant(1.2, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])

f_predict = loaded_model.predict(x, t)
g = tf.reshape(exact(x, t), (num_points, num_points))
g_nn = tf.reshape(f_predict, (num_points, num_points))

G = g.numpy()
G_NN = g_nn.numpy()
G = G[:,1:-1]
G_NN = G_NN[:,1:-1]

rel_error = np.abs(G_NN - G)/G
rel_x = np.linspace(0,1,num_points)
rel_x = rel_x[1:-1]
rel_t = np.linspace(1,1.2,num_points)


RX,RT = np.meshgrid(rel_x,rel_t)

plt.contourf(RX, RT, rel_error, levels=41, cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
cb = plt.colorbar()
cb.set_label(label="Relative Error", size=fontsize)
plt.savefig("../results/neural_net/pde/pde_rel_error_outside_training_domain.pdf")
plt.show()

G = g.numpy().ravel()
G_NN = g_nn.numpy().ravel()
res = np.sum((G-G_NN)**2)
tot = np.sum((G - np.mean(G))**2)
R2 = 1 - res/tot
print("R2 ONLY outside domain= ", R2)

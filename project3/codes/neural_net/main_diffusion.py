import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neural_diffusion_solver import NeuralDiffusionSolver

seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

exact = lambda x, t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)

layers = [1000, 1000, 1000, 1000, 1]
input_sz = 2
n = 100

x = np.random.uniform(0, 1, n)
t = np.random.uniform(0, 1, n)

epochs = 10

my_model = NeuralDiffusionSolver(layers=layers, input_sz=input_sz)
epoch_arr, loss = my_model.fit(x=x, t=t, epochs=epochs)

"""
fontsize = 16


plt.plot(epoch_arr, loss)
plt.xlabel("Epochs", size=fontsize-2)
plt.ylabel("Loss", size=fontsize-2)
plt.xticks(fontsize=fontsize-2)
plt.yticks(fontsize=fontsize-2)
plt.savefig("../results/neural_net/pde/epoch_vs_loss_optimal_params.pdf")
plt.show()

#ON T: 0 - 1
num_points = 40
start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(1, dtype=tf.float32)
start_t = tf.constant(0, dtype=tf.float32)
stop_t = tf.constant(1, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])


f_predict = my_model.predict(x, t)
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

fig = plt.figure()
ax = fig.add_subplot(111)
plt.pcolormesh(X, T, g,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.set_label(label=r"$u(x,t)$", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
plt.savefig("../results/neural_net/pde/pde_analytical_on_training_domain.pdf")

fig1 = plt.figure()
ax = fig1.add_subplot(111)
plt.pcolormesh(X, T, g_nn,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.set_label(label=r"$u(x,t)$", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
plt.savefig("../results/neural_net/pde/pde_network_on_training_domain.pdf")
plt.show()
<<<<<<< HEAD
"""
=======

fig2 = plt.figure()
ax = fig2.add_subplot(111)
plt.pcolormesh(RX, RT, rel_error,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.set_label(label="Relative error", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
plt.savefig("../results/neural_net/pde/pde_rel_error_on_training_domain.pdf")
plt.show()

G = g.numpy().ravel()
G_NN = g_nn.numpy().ravel()
res = np.sum((G-G_NN)**2)
tot = np.sum((G - np.mean(G))**2)
R2 = 1 - res/tot
print("R2 on domain = ", R2)




#ON T: 0 - 1.2
start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(1, dtype=tf.float32)
start_t = tf.constant(1, dtype=tf.float32)
stop_t = tf.constant(1.2, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])
>>>>>>> 2d849c8cafb148d000ca1a34663e1e48d1070725

f_predict = my_model.predict(x, t)
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

fig = plt.figure(figsize = (9,5))
ax = fig.add_subplot(111)
plt.pcolormesh(X, T, g,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.get_offset_text().set_fontsize(fontsize)
cb.set_label(label=r"$u(x,t)$", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
cb.update_ticks()
plt.savefig("../results/neural_net/pde/pde_analytical_outside_training_domain.pdf")

fig2 = plt.figure(figsize = (9,5))
ax = fig2.add_subplot(111)
plt.pcolormesh(X, T, g_nn,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.get_offset_text().set_fontsize(fontsize)
cb.set_label(label=r"$u(x,t)$", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
cb.update_ticks()
plt.savefig("../results/neural_net/pde/pde_network_outside_training_domain.pdf")
plt.show()

fig3 = plt.figure(figsize = (9,5))
ax = fig3.add_subplot(111)
plt.pcolormesh(RX, RT, rel_error,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.get_offset_text().set_fontsize(fontsize)
cb.set_label(label="Relative error", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
cb.update_ticks()
plt.savefig("../results/neural_net/pde/pde_rel_error_outside_training_domain.pdf")
plt.show()


G = g.numpy().ravel()
G_NN = g_nn.numpy().ravel()
res = np.sum((G-G_NN)**2)
tot = np.sum((G - np.mean(G))**2)
R2 = 1 - res/tot
print("R2 ONLY outside domain= ", R2)




#ON T: 0 - 0.5
num_points = 40
start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(1, dtype=tf.float32)
start_t = tf.constant(0, dtype=tf.float32)
stop_t = tf.constant(0.5, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])


f_predict = my_model.predict(x, t)
g = tf.reshape(exact(x, t), (num_points, num_points))
g_nn = tf.reshape(f_predict, (num_points, num_points))

G = g.numpy()
G_NN = g_nn.numpy()
G = G[1:,1:-1]
G_NN = G_NN[1:,1:-1]

rel_error = np.abs(G_NN - G)/G
rel_x = np.linspace(0,1,num_points)
rel_x = rel_x[1:-1]
rel_t = np.linspace(0,0.5,num_points)
rel_t = rel_t[1:]


RX,RT = np.meshgrid(rel_x,rel_t)

fig = plt.figure(figsize = (9,5))
ax = fig.add_subplot(111)
plt.pcolormesh(X, T, g,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.set_label(label=r"$u(x,t)$", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
plt.savefig("../results/neural_net/pde/pde_analytical_HALF_training_domain.pdf")

fig1 = plt.figure(figsize = (9,5))
ax = fig1.add_subplot(111)
plt.pcolormesh(X, T, g_nn,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.set_label(label=r"$u(x,t)$", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
plt.savefig("../results/neural_net/pde/pde_network_HALF_training_domain.pdf")
plt.show()

fig2 = plt.figure(figsize = (9,5))
ax = fig2.add_subplot(111)
plt.pcolormesh(RX, RT, rel_error,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.set_label(label="Relative error", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
plt.savefig("../results/neural_net/pde/pde_rel_error_HALF_training_domain.pdf")
plt.show()

G = g.numpy().ravel()
G_NN = g_nn.numpy().ravel()
res = np.sum((G-G_NN)**2)
tot = np.sum((G - np.mean(G))**2)
R2 = 1 - res/tot
print("R2 on HALF domain = ", R2)

#ON T: 0 - 0.2
num_points = 40
start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(1, dtype=tf.float32)
start_t = tf.constant(0, dtype=tf.float32)
stop_t = tf.constant(0.2, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])


f_predict = my_model.predict(x, t)
g = tf.reshape(exact(x, t), (num_points, num_points))
g_nn = tf.reshape(f_predict, (num_points, num_points))

G = g.numpy()
G_NN = g_nn.numpy()
G = G[1:,1:-1]
G_NN = G_NN[1:,1:-1]

rel_error = np.abs(G_NN - G)/G
rel_x = np.linspace(0,1,num_points)
rel_x = rel_x[1:-1]
rel_t = np.linspace(0,0.2,num_points)
rel_t = rel_t[1:]


RX,RT = np.meshgrid(rel_x,rel_t)

fig = plt.figure(figsize = (9,5))
ax = fig.add_subplot(111)
plt.pcolormesh(X, T, g,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.set_label(label=r"$u(x,t)$", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
plt.savefig("../results/neural_net/pde/pde_analytical_p2_training_domain.pdf")

<<<<<<< HEAD
fontsize = 16
ticksize = 16

plt.contourf(X, T, rel_err, cmap="inferno", levels=1000)
cbar = plt.colorbar()
cbar.set_label("Relative Error", size=fontsize)
cbar.ax.tick_params(labelsize=ticksize)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()
plt.xticks(size=ticksize)
plt.yticks(size=ticksize)
ax.set_xlabel(r"$x$", size=fontsize)
ax.set_ylabel(r"$t$", size=fontsize)
plt.savefig("test1.pdf", dpi=1000)
=======
fig1 = plt.figure(figsize = (9,5))
ax = fig1.add_subplot(111)
plt.pcolormesh(X, T, g_nn,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.set_label(label=r"$u(x,t)$", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
plt.savefig("../results/neural_net/pde/pde_network_p2_training_domain.pdf")
plt.show()

fig2 = plt.figure(figsize = (9,5))
ax = fig2.add_subplot(111)
plt.pcolormesh(RX, RT, rel_error,  cmap="inferno")
plt.xlabel("x", size=fontsize)
plt.ylabel("t", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cb = plt.colorbar()
cb.set_label(label="Relative error", size=fontsize)
cb.ax.tick_params(labelsize = fontsize)
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.get_offset_text().set_fontsize(fontsize)
cb.update_ticks()
plt.savefig("../results/neural_net/pde/pde_rel_error_p2_training_domain.pdf")
>>>>>>> 2d849c8cafb148d000ca1a34663e1e48d1070725
plt.show()

G = g.numpy().ravel()
G_NN = g_nn.numpy().ravel()
res = np.sum((G-G_NN)**2)
tot = np.sum((G - np.mean(G))**2)
R2 = 1 - res/tot
print("R2 on p2 domain = ", R2)

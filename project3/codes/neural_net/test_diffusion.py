import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neural_diffusion_solver import NeuralDiffusionSolver


seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

exact = lambda x, t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)


n = 10
x = np.linspace(0, 1, n)
t = np.linspace(0, 1, n)
x, t = np.meshgrid(x, t)
x, t = x.ravel(), t.ravel()

#Fit the model
layers = [100]*4 + [1]
input_sz = 2
epochs = 1000
my_model = NeuralDiffusionSolver(layers=layers, input_sz=input_sz, learning_rate=0.001)
print("\n\n\n\n\n\n\n")
epoch_arr, loss = my_model.fit(x=x, t=t, epochs=epochs)

#Plot loss vs epochs
plt.plot(epoch_arr, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# Define grid and compute predicted solution
num_points = 40
start = tf.constant(0.01, dtype=tf.float32)
stop = tf.constant(0.99, dtype=tf.float32)
start_t = tf.constant(0, dtype=tf.float32)
stop_t = tf.constant(0.5, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])
f_predict = my_model.predict(x, t)
g = tf.reshape(exact(x, t), (num_points, num_points))
g_nn = tf.reshape(f_predict, (num_points, num_points))
rel_err = np.abs((g - g_nn)/g)



#Plot relative error
fig = plt.figure()
ax = fig.add_subplot(111)
fontsize = 16
ticksize = 16
plt.pcolormesh(X, T, rel_err, cmap="inferno")
cbar = plt.colorbar()
cbar.set_label("Relative error", size=fontsize)
cbar.ax.tick_params(labelsize=ticksize)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()
plt.xticks(size=ticksize)
plt.yticks(size=ticksize)
ax.set_xlabel(r"$x$", size=fontsize)
ax.set_ylabel(r"$t$", size=fontsize)

#Plot solution
plt.figure()
plt.pcolormesh(X, T, g_nn, cmap="inferno")
cbar = plt.colorbar()
cbar.set_label("u(x,t)", size=fontsize)
cbar.ax.tick_params(labelsize=ticksize)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()
plt.xticks(size=ticksize)
plt.yticks(size=ticksize)
ax.set_xlabel(r"$x$", size=fontsize)
ax.set_ylabel(r"$t$", size=fontsize)
plt.show()

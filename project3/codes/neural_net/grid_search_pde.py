import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neural_diffusion_solver import NeuralDiffusionSolver

# seed = 10
# tf.random.set_seed(seed)
# np.random.seed(seed)

exact = lambda x, t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)
input_sz = 2
n = 100
x_train = np.random.uniform(0, 1, n)
t_train = np.random.uniform(0, 1, n)
epochs = 500

# Define grid
num_points = 41
start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(1, dtype=tf.float32)
start_t = tf.constant(0, dtype=tf.float32)
stop_t = tf.constant(1, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])

num_layers = [2,4,6,8,10]
num_nodes = [10,50,100,500,1000]

R2_mat = np.zeros([len(num_layers), len(num_nodes)])

for l in num_layers:
    for no in num_nodes:
        layers = [no]*l + [1]
        my_model = NeuralDiffusionSolver(layers, input_sz)
        my_model.fit(x=x_train, t=t_train, epochs=epochs)
        f_predict = my_model.predict(x, t)

        g = tf.reshape(exact(x, t), (num_points, num_points))
        g_nn = tf.reshape(f_predict, (num_points, num_points))

        G = g.numpy().ravel()
        G_NN = g_nn.numpy().ravel()
        res = np.sum((G-G_NN)**2)
        tot = np.sum((G - np.mean(G))**2)
        R2 = 1 - res/tot
        i = num_layers.index(l)
        j = num_nodes.index(no)
        R2_mat[i,j] = R2

        print("R2 = ", R2)

np.save('grid_search_R2_pde.npy', R2_mat)

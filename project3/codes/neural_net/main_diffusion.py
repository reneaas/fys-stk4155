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

epochs = 500

my_model = NeuralDiffusionSolver(layers, input_sz)
epoch_arr, loss = my_model.fit(x=x, t=t, epochs=epochs)

my_model.save('my_pde_model')

fontsize = 14
plt.plot(epoch_arr, loss)
plt.xlabel("epochs", size=fontsize)
plt.ylabel("loss", size=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.savefig("../results/neural_net/pde/epoch_vs_loss_optimal_params.pdf")
plt.show()

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow_addons as tfa

#tf.keras.backend.set_floatx("float64")
np.random.seed(100)
tf.random.set_seed(100)


def create_DNN(layers, input_size):
    """
    Input:
    ------
    layers: list
        containing the number of nodes in each layer.
    input_size: int or tuple
    """

    # Set up model
    model = tf.keras.Sequential()

    # First hidden layer
    model.add(tf.keras.layers.Dense(layers[0], input_shape=(input_size,), activation=None))

    # Hidden layers
    for layer in layers[1:-1]:
        model.add(tf.keras.layers.Dense(layer, activation="relu"))
        #model.add(tf.keras.layers.Dropout(0.1))

    # Output layer
    model.add(tf.keras.layers.Dense(layers[-1], activation="linear"))

    return model


@tf.function
def trial_function(x, t, model, training):
    X = tf.concat([x,t],1)
    return tf.sin(np.pi*x) + t*x*(1-x)*model(X, training=training)



@tf.function
def loss(x, t, y, trial_function):
    with tf.GradientTape() as tape:
        tape.watch(x)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x,t])

            f_trial = trial_function(x, t, model, True)

        # Calculcating derivatives
        df_dt = tape1.gradient(f_trial, t)
        df_dx = tape1.gradient(f_trial, x)

    d2f_dx2 = tape.gradient(df_dx, x)

    del tape
    del tape1

    y_pred = df_dt - d2f_dx2
    loss_value = loss_fn(y, y_pred)

    return loss_value


@tf.function
def grad(x, t, y, trial_function):
    with tf.GradientTape() as tape:
        loss_value = loss(x, t, y, trial_function)
    return loss_value, tape.gradient(loss_value, model.trainable_weights)


@tf.function
def predict(x, t):
    X = tf.concat([x,t],1)
    N = model(X, training=False)
    g_trial = tf.sin(np.pi*x) + t*x*(1-x)*N
    return g_trial


layers = [10, 1000, 250, 20, 10, 1]
model = create_DNN(layers, input_size=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
loss_fn = tf.keras.losses.MeanSquaredError()




n = 100
x = np.random.uniform(0, 1, n)
t = np.random.uniform(0, 1, n)

x = tf.convert_to_tensor(x.reshape(-1,1), dtype=tf.float32)
t = tf.convert_to_tensor(t.reshape(-1,1), dtype=tf.float32)

zeros = np.zeros_like(x)
ground_truth = tf.convert_to_tensor(zeros, dtype=tf.float32)

epochs = 500
for i in range(epochs):
    loss_value, gradients = grad(x, t, ground_truth, trial_function)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print("loss: ", loss_value.numpy())


# Define grid
num_points = 41

start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(1, dtype=tf.float32)
start_t = tf.constant(0, dtype=tf.float32)
stop_t = tf.constant(1, dtype=tf.float32)

X, T = tf.meshgrid(tf.linspace(start_t, stop_t, num_points), tf.linspace(start_t, stop_t, num_points))

x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])


"""
n = 1000
x = np.linspace(-1,2, n)
X = tf.convert_to_tensor(x.reshape(-1,1), dtype=tf.float32)
t = np.ones(n)*0.1
T = tf.convert_to_tensor(t.reshape(-1,1), dtype=tf.float32)


g_predict = predict(X, T)

exact = lambda x, t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)

plt.plot(x, exact(x,0.01), label="exact")
plt.plot(x, g_predict, label="dnn")
plt.legend()
plt.show()
"""


exact = lambda x, t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)

# Plot solution on larger grid
num_points = 41
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])

g_predict = predict(x, t)

g = tf.reshape(exact(x, t), (num_points, num_points))
g_nn = tf.reshape(g_predict, (num_points, num_points))

diff = tf.abs(g - g_nn)
print(f"Max diff: {tf.reduce_max(diff)}")
print(f"Mean diff: {tf.reduce_mean(diff)}")

G = g.numpy().ravel()
G_NN = g_nn.numpy().ravel()
print(G)
print(G_NN)
res = np.sum((G-G_NN)**2)
tot = np.sum((G - np.mean(G))**2)
R2 = 1 - res/tot
print("R2 = ", R2)


#fig = plt.figure()
#ax = fig.gca(projection="3d")
#ax.set_title("Analytic")
#ax.plot_surface(X, T, g)
plt.contourf(X,T, g, levels=41)
plt.colorbar()

fig = plt.figure()
#ax = fig.gca(projection="3d")
#ax.set_title("Neural")
#ax.plot_surface(X, T, g_nn)
plt.contourf(X,T, g_nn, levels=41)
plt.colorbar()

"""
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_title("Diff")
ax.plot_surface(X, T, diff)
"""

plt.show()

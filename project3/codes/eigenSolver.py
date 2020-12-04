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
def trial_function(t, x0, model, training):
    return tf.exp(-t)*x0 + (1-tf.exp(-t))*model(t, training=training)



@tf.function
def loss(t, x0, A, y, trial_function):
    with tf.GradientTape() as tape:
        tape.watch(t)

        x_trial = trial_function(t, x0, model, True)
        print("XTRIAL = ", x_trial.get_shape())

    dx_dt = tape.gradient(x_trial, t)

    del tape

    shape = A.get_shape().as_list()
    I = tf.eye(shape[0])

    term1 = tf.transpose(x_trial)*x_trial*A
    print("TERM1 = ", term1.get_shape())
    term2 = (1 - tf.transpose(x_trial)*tf.linalg.matmul(A,x_trial) )*I
    print("TERM2 = ", term2.get_shape())
    f = tf.linalg.matmul((term1 + term2),x_trial)
    print("F = ", f.get_shape())

    y_pred = dx_dt + x_trial - f
    loss_value = loss_fn(y, y_pred)

    return loss_value


@tf.function
def grad(t, x0, A, y, trial_function):
    with tf.GradientTape() as tape:
        loss_value = loss(t, x0, A, y, trial_function)
    return loss_value, tape.gradient(loss_value, model.trainable_weights)


@tf.function
def predict(t, x0):
    N = model(t, training=False)
    g_trial = tf.exp(-t)*x0 + (1-tf.exp(-t))*model(t, training=False)
    return g_trial


layers = [10, 100, 200, 1000, 250, 20, 10, 1]
model = create_DNN(layers, input_size=1)

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()


n = 3

t = np.linspace(0, 1, n)
x0 = np.array([1.1,0.1,0.1])

x0 = tf.convert_to_tensor(x0.reshape(-1,1), dtype=tf.float32)
t = tf.convert_to_tensor(t.reshape(-1,1), dtype=tf.float32)


#Anp = np.array([[2, 4, 4], [4, 8, 8], [4, 8, 8]])
Anp = np.array([[3, 0, 4], [0, 2, 0], [4, 0, 3]])
A = tf.convert_to_tensor(Anp.reshape(3,3), dtype=tf.float32)

zeros = np.zeros_like(t)
ground_truth = tf.convert_to_tensor(zeros, dtype=tf.float32)

epochs = 500
for i in range(epochs):
    loss_value, gradients = grad(t, x0, A, ground_truth, trial_function)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print("loss: ", loss_value.numpy())


eigen_vec_predict = predict(x0, t)
eig_vec = eigen_vec_predict.numpy()

true_val = np.linalg.eig(Anp)
print("TRUE = ", true_val)

eig_val = (eig_vec.T@Anp@eig_vec)/eig_vec.T@eig_vec

print(eig_val)

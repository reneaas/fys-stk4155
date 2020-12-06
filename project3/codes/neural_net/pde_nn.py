import tensorflow as tf
import numpy as np
from progress.bar import Bar

seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

class PDE_NN(tf.keras.Sequential):
    def __init__(self, layers, input_sz):
        super(PDE_NN, self).__init__()
        # Set up model
        # First hidden layer connected to the input
        self.add(tf.keras.layers.Dense(layers[0], input_shape=(input_sz,), activation=None))

        # Hidden layers
        for layer in layers[1:-1]:
            self.add(tf.keras.layers.Dense(layer, activation="relu"))
            #self.add(tf.keras.layers.Dropout(0.1))

        # Output layer
        self.add(tf.keras.layers.Dense(layers[-1], activation="linear"))

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()


    #@tf.function
    def train(self, x, t, epochs):
        x = x.reshape(-1,1)
        t = t.reshape(-1,1)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        self.x = x
        self.t = t

        bar = Bar("Epochs", max = epochs)
        for epoch in range(epochs):
            bar.next()
            loss, gradients = self.compute_gradients()
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        bar.finish()
        return None

    @tf.function
    def predict(self, x, t):
        self.x = x
        self.t = t
        f_trial = self.trial_function(training=False)
        return f_trial


    @tf.function
    def trial_function(self,training):
        x, t = self.x, self.t
        X = tf.concat([x,t], 1)
        N = self(X, training=training)
        f_trial = tf.sin(np.pi*x) + t*x*(1-x)*N
        return f_trial

    @tf.function
    def compute_loss(self):
        #Compute gradients for G = d^2f/dx^2 - df/dt = 0
        x, t = self.x, self.t
        with tf.GradientTape() as gg:
            gg.watch(x)
            with tf.GradientTape(persistent=True) as g:
                g.watch([x,t])
                f_trial = self.trial_function(training=True)

            df_dt = g.gradient(f_trial, t)
            df_dx = g.gradient(f_trial, x)

        d2f_dx2 = gg.gradient(df_dx, x)

        #Delete references
        del g
        del gg

        #Compute "prediction" i.e cost/loss-function and loss value.
        y_pred = df_dt - d2f_dx2
        loss = self.loss_fn(0., y_pred)
        return loss

    @tf.function
    def compute_gradients(self):
        with tf.GradientTape() as tape:
            loss = self.compute_loss()
        gradients = tape.gradient(loss, self.trainable_variables)
        return loss, gradients

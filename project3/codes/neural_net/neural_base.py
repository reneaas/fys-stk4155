import tensorflow as tf
import numpy as np

class NeuralBase(tf.keras.Sequential):
    def __init__(self, layers, input_sz, learning_rate=0.001):
        super(NeuralBase, self).__init__()
        # Set up model
        # First hidden layer connected to the input
        self.add(tf.keras.layers.Dense(layers[0], input_shape=(input_sz,), activation="linear"))

        # Hidden layers
        for layer in layers[1:-1]:
            self.add(tf.keras.layers.Dense(layer, activation="sigmoid"))
            # self.add(tf.keras.layers.Dropout(0.1))

        # Output layer
        self.add(tf.keras.layers.Dense(layers[-1], activation="linear"))

        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.7)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def compute_gradients(self):
        with tf.GradientTape() as tape:
            loss = self.compute_loss()
        gradients = tape.gradient(loss, self.trainable_variables)
        return loss, gradients

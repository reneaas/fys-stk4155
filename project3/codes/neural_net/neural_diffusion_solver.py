import tensorflow as tf
import numpy as np
from neural_base import NeuralBase
from progress.bar import Bar

seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

class NeuralDiffusionSolver(NeuralBase):
    def __init__(self, layers, input_sz, learning_rate):
        super(NeuralDiffusionSolver, self).__init__(layers, input_sz, learning_rate)


    def fit(self, x, t, epochs):
        x = x.reshape(-1,1)
        t = t.reshape(-1,1)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        self.x = x
        self.t = t


        epoch_arr = np.zeros(epochs)
        losses = np.zeros(epochs)

        bar = Bar("Epochs", max = epochs)
        for epoch in range(epochs):
            bar.next()

            loss, gradients = self.compute_gradients()
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            epoch_arr[epoch] = epoch
            losses[epoch] = loss

        bar.finish()
        return epoch_arr, losses


    @tf.function
    def predict(self, x, t):
        self.x = x
        self.t = t
        f_trial = self.trial_fn(training=False)
        return f_trial


    @tf.function
    def trial_fn(self,training):
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
                f_trial = self.trial_fn(training=True)

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

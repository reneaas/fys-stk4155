import tensorflow as tf
import numpy as np
from neural_net import NeuralNet
from progress.bar import Bar

seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)


class EigenSolver_NN(NeuralNet):
    def __init__(self, layers, input_sz, matrix):
        super(EigenSolver_NN, self).__init__(layers, input_sz)
        self.A = tf.convert_to_tensor(matrix, dtype=tf.float32)
        self.mat_sz = layers[-1]
        self.I = tf.eye(num_rows=self.mat_sz, dtype=tf.float32)

    def fit(self, x, t, epochs):
        #Set up meshgrid
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        x, t = tf.meshgrid(x, t)
        x, t = tf.reshape(x, [-1, 1]), tf.reshape(t, [-1, 1])
        self.x, self.t = x, t

        #fit the model
        bar = Bar("Epochs", max = epochs)
        for epoch in range(epochs):
            bar.next()
            loss, gradients = self.compute_gradients()
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            #print(f"iteration: {epoch} ; loss: {loss.numpy()}")
        bar.finish()
        return None


    @tf.function
    def trial_function(self, x, t, training):
        N = self(t, training=training)
        return tf.exp(-t)*x + (1-tf.exp(-t))*N


    @tf.function
    def compute_loss(self):
        with tf.GradientTape() as tape:
            tape.watch(self.t)
            self.x_trial = self.trial_function(self.x, self.t, training=True)
        dx_dt = tape.batch_jacobian(self.x_trial, self.t)
        del tape

        dx_dt = tf.transpose(tf.reduce_sum(dx_dt, axis=2))
        x_trial = tf.transpose(self.x_trial)

        xx = tf.reduce_sum(tf.multiply(x_trial, x_trial))
        Ax = tf.matmul(self.A, x_trial)
        xAx = tf.reduce_sum(tf.multiply(x_trial, Ax))
        M = (xx*self.A + (1-xAx)*self.I)
        f = tf.matmul(M, x_trial)
        y_pred = dx_dt + x_trial - f
        loss = self.loss_fn(0., y_pred)
        return loss


    def eig(self, x, t):
        """
        returns eigenvalue and corresponding normalized eigenvector
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        v = self.trial_function(x, t, training=False)
        v = tf.transpose(v) #Must be transposed to get the correct mathematical shape
        Av = tf.matmul(self.A, v)
        vAv = tf.reduce_sum(tf.multiply(v, Av))
        vv = tf.reduce_sum(tf.multiply(v,v))
        eigenvalue = vAv/vv
        v_norm = tf.sqrt(vv)
        return eigenvalue, v/v_norm

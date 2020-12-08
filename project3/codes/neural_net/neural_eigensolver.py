import tensorflow as tf
import numpy as np
from neural_base import NeuralBase
from progress.bar import Bar

seed = 10
# tf.random.set_seed(seed)
# np.random.seed(seed)


class NeuralEigenSolver(NeuralBase):
    def __init__(self, layers, input_sz, matrix, eig_type):
        super(NeuralEigenSolver, self).__init__(layers, input_sz)

        if eig_type == "max":
            self.A_train = tf.convert_to_tensor(matrix, dtype=tf.float32)
        elif eig_type == "min":
            self.A_train = tf.convert_to_tensor(-matrix, dtype=tf.float32)

        self.A_test = tf.convert_to_tensor(matrix, dtype=tf.float32)

        self.mat_sz = layers[-1]
        self.I = tf.eye(num_rows=self.mat_sz, dtype=tf.float32)

    def fit(self, x, t, epochs):
        #Set up meshgrid
        x0 = x #start vector
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        x, t = tf.meshgrid(x, t)
        x, t = tf.reshape(x, [-1, 1]), tf.reshape(t, [-1, 1])
        self.x, self.t = x, t

        #Arrays to store predictions as function of epochs
        epoch_arr = np.zeros(epochs)
        eigvals = np.zeros(epochs)
        eigvecs = np.zeros([epochs, self.mat_sz])
        t_max = tf.reshape(self.t[-1], [-1,1])

        #Fit the model
        bar = Bar("Epochs", max = epochs)
        for epoch in range(epochs):
            bar.next()
            loss, gradients = self.compute_gradients()
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            eigval, eigvec = self.eig(x0, t_max, self.A_test)
            eigvals[epoch] = eigval.numpy()
            eigvecs[epoch,:] = eigvec.numpy().T[:]
            epoch_arr[epoch] = epoch
        bar.finish()
        return epoch_arr, eigvals, eigvecs


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
        Ax = tf.matmul(self.A_train, x_trial)
        xAx = tf.reduce_sum(tf.multiply(x_trial, Ax))
        M = (xx*self.A_train + (1-xAx)*self.I)
        f = tf.matmul(M, x_trial)
        y_pred = dx_dt + x_trial - f
        loss = self.loss_fn(0., y_pred)
        return loss


    def eig(self, x, t, A):
        """
        returns eigenvalue and corresponding normalized eigenvector
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        v = self.trial_function(x, t, training=False)
        v = tf.transpose(v) #Must be transposed to get the correct mathematical shape
        Av = tf.matmul(self.A_test, v)
        vAv = tf.reduce_sum(tf.multiply(v, Av))
        vv = tf.reduce_sum(tf.multiply(v,v))
        eigval = vAv/vv
        v_norm = tf.sqrt(vv)
        eigvec = v/v_norm
        return eigval, eigvec

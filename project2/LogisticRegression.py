import numpy as np
from progress.bar import Bar
np.random.seed(1001)

class LogReg():
    def __init__(self, classes, N, X_data, y_data, eta = 0.1, gamma = 0, Lambda = 0, N_outputs, epochs = 10):

        self.M = classes
        self.eta = eta
        self.epochs = epochs

        self.X_data = X_data
        self.y_data = y_data
        self.N_points, self.features = np.shape(X_data)

        self.Lambda = Lambda
        selg.gamma = gamma

        self.grad_weights = np.zeros([self.M,self.features])
        self.weights = np.random.normal(size=[self.M,self.features])

        self.bias = np.random.normal(size=[self.M,self.features])
        self.grad_bias = np.zeros([self.M,self.features])

        self.error = np.zeros([self.M,self.features])
        self.output = np.zeros(self.M)

        self.tmp_weights = np.zeros([self.M,self.features])
        self.tmp_bias = np.zeros([self.M,self.features])

    def train(self):
        batches = self.N_points//self.batch_size
        total_indices = np.arange(self.N_points)

        bar = Bar("Epoch ", max = self.epochs)
        for epoch in range(self.epochs):
            bar.next()
            for b in range(batches):
                indices = np.random.choice(total_indices, size=self.batch_size, replace=True)
                self.X  = self.X_data[indices]
                self.y = self.y_data[indices]
                for i in range(self.batch_size):
                    z = (self.weights @ self.X[i]) + self.bias
                    self.output = self.softmax(z)

                    self.error = self.compute_error_output(self.output, self.y[i])
                    self.grad_bias += self.error
                    self.grad_weights += np.outer(self.error, self.output) + self.Lambda*self.weights

                self.update_parameters()
        bar.finish()

    def predict(self,x):
        z = (self.weights @ x) + self.bias
        self.output = self.softmax(z)
        return self.output



    def update_parameters(self):
        scale = self.eta/self.batch_size
        self.grad_bias *= scale
        self.grad_weights *= scale

        self.tmp_weights = (self.grad_weights + self.gamma*self.tmp_weights)
        self.weights -= self.tmp_weights

        self.tmp_bias = (self.grad_bias + self.gamma*self.tmp_bias)
        self.bias -= self.tmp_bias

        self.grad_bias[:,:] = 0.
        self.grad_weights[:,:] = 0.



    @staticmethod
    def softmax(z):
        Z = np.sum(np.exp(z))
        return np.exp(z)/Z

    @staticmethod
    def compute_error_output(output, y):
        return output - y

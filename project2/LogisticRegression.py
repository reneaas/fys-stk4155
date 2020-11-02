import numpy as np
from progress.bar import Bar
np.random.seed(1001)

class LogReg():
    def __init__(self, classes, X_data, y_data, eta, gamma, Lambda, epochs, batch_size):

        self.M = classes
        self.eta = eta
        self.epochs = epochs

        self.X_data = X_data
        self.y_data = y_data
        self.N_points, self.features = np.shape(X_data)

        self.Lambda = Lambda
        self.gamma = gamma
        self.batch_size = batch_size

        self.grad_weights = np.zeros([self.M,self.features])
        self.weights = np.random.normal(size=[self.M,self.features])*0.001

        self.bias = np.random.uniform(size=self.M)
        self.grad_bias = np.zeros(self.M)

        self.output = np.zeros(self.M)

        self.tmp_weights = np.zeros([self.M,self.features])
        self.tmp_bias = np.zeros(self.M)

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
                    self.grad_weights += self.compute_grad_weights(self.y[i], self.X[i]) + self.Lambda*self.weights
                    self.grad_bias += self.compute_grad_bias(self.y[i])
                self.update_parameters()
        bar.finish()

    def predict(self,x):
        z = (self.weights @ x) + self.bias
        self.output = self.softmax(z)
        return self.output



    def update_parameters(self):
        scale = self.eta/self.batch_size
        self.grad_weights *= scale
        self.grad_bias *= scale

        self.tmp_weights = (self.grad_weights + self.gamma*self.tmp_weights)
        self.weights -= self.tmp_weights

        self.tmp_bias = (self.grad_bias + self.gamma*self.tmp_bias)
        self.bias -= self.tmp_bias

        self.grad_weights[:,:] = 0.
        self.grad_bias[:] = 0.


    @staticmethod
    def softmax(z):
        Z = np.sum(np.exp(z))
        return np.exp(z)/Z


    def compute_grad_weights(self, y, x):
        return np.outer(self.output-y, x)

    def compute_grad_bias(self, y):
        return self.output - y

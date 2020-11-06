import numpy as np
from progress.bar import Bar
np.random.seed(1001)

class LogReg():
    def __init__(self, classes, X_data, y_data, eta, gamma, Lambda, epochs, batch_size, optimizer):

        self.M = classes
        self.epochs = epochs

        self.X_data = X_data
        self.y_data = y_data
        self.N_points, self.features = np.shape(X_data)

        self.Lambda = Lambda
        self.batch_size = batch_size
        self.eta = eta/self.batch_size

        self.grad_weights = np.zeros([self.M,self.features])
        self.weights = np.random.normal(size=[self.M,self.features])/np.sqrt(self.features)

        self.bias = np.random.uniform(size=self.M)
        self.grad_bias = np.zeros(self.M)

        self.output = np.zeros(self.M)

        if optimizer == "SGD momentum":
            self.tmp_weights = np.zeros([self.M,self.features])
            self.tmp_bias = np.zeros(self.M)
            self.gamma = gamma
            self.optimizer = lambda: self.SGD_with_momentum()

        if optimizer == "SGD":
            self.optimizer = lambda: self.SGD()

        if optimizer == "ADAM":
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.epsilon = 10**(-2)

            self.momentum_weights = np.zeros([self.M,self.features])
            self.momentum_bias = np.zeros(self.M)

            self.second_momentum_weights = np.zeros([self.M,self.features])
            self.second_momentum_bias = np.zeros(self.M)

            self.scaled_momentum_weights = np.zeros([self.M,self.features])
            self.scaled_momentum_bias = np.zeros(self.M)

            self.scaled_second_momentum_weights = np.zeros([self.M,self.features])
            self.scaled_second_momentum_bias = np.zeros(self.M)

            self.optimizer = lambda: self.ADAM()

    def train(self):
        batches = self.N_points//self.batch_size
        total_indices = np.arange(self.N_points)

        bar = Bar("Epoch ", max = self.epochs)
        for epoch in range(self.epochs):
            bar.next()
            for self.b in range(batches):
                indices = np.random.choice(total_indices, size=self.batch_size, replace=True)
                self.X  = self.X_data[indices]
                self.y = self.y_data[indices]
                for i in range(self.batch_size):
                    z = (self.weights @ self.X[i]) + self.bias
                    self.output = self.softmax(z)
                    self.grad_weights += self.compute_grad_weights(self.y[i], self.X[i]) + self.Lambda*self.weights
                    self.grad_bias += self.compute_grad_bias(self.y[i])
                self.optimizer()
        bar.finish()

    def predict(self,x):
        z = (self.weights @ x) + self.bias
        self.output = self.softmax(z)
        return self.output


    def SGD(self):
        scale = self.eta
        self.grad_weights *= self.eta
        self.grad_bias *= self.eta

        self.weights -= self.grad_weights
        self.bias -= self.grad_bias

        self.grad_weights[:,:] = 0.
        self.grad_bias[:] = 0.

    def SGD_with_momentum(self):
        self.grad_weights *= self.eta
        self.grad_bias *= self.eta

        self.tmp_weights = (self.grad_weights + self.gamma*self.tmp_weights)
        self.weights -= self.tmp_weights

        self.tmp_bias = (self.grad_bias + self.gamma*self.tmp_bias)
        self.bias -= self.tmp_bias

        self.grad_weights[:,:] = 0.
        self.grad_bias[:] = 0.

    def ADAM(self):
        self.grad_weights *= self.eta
        self.grad_bias *= self.eta

        self.momentum_weights = self.beta1*self.momentum_weights + (1-self.beta1)*self.grad_weights
        self.momentum_bias = self.beta2*self.momentum_bias + (1-self.beta1)*self.grad_bias

        self.second_momentum_weights = self.beta2*self.second_momentum_weights + (1-self.beta2)*(self.grad_weights*self.grad_weights)
        self.second_momentum_bias = self.beta2*self.second_momentum_bias + (1-self.beta2)*(self.grad_bias*self.grad_bias)

        self.scaled_momentum_weights = self.momentum_weights/(1-self.beta1**(self.b+1))
        self.scaled_momentum_bias = self.momentum_bias/(1-self.beta1**(self.b+1))

        self.scaled_second_momentum_weights = self.second_momentum_weights/(1-self.beta2**(self.b+1))
        self.scaled_second_momentum_bias = self.second_momentum_bias/(1-self.beta2**(self.b+1))

        self.weights -= self.scaled_momentum_weights/(np.sqrt(self.scaled_second_momentum_weights) + self.epsilon)
        self.bias -= self.scaled_momentum_bias/(np.sqrt(self.scaled_second_momentum_bias) + self.epsilon)

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

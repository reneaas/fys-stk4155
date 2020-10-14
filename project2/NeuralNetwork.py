import numpy as np
from progress.bar import Bar
np.random.seed(1001)

class FFNN():

    def __init__(self, layers, nodes, X_data, y_data, M_outputs, epochs=10, batch_size=100, eta = 0.51, problem_type="classification", hidden_activation="sigmoid"):
        self.layers = layers
        self.nodes = nodes
        self.X_data = X_data
        self.y_data = y_data
        self.N_points, self.features = np.shape(X_data)
        self.M_outputs = M_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta

        #self.error_input = np.zeros(self.nodes)
        self.error_hidden = np.zeros([self.layers,self.nodes])
        self.error_output = np.zeros(self.M_outputs)

        #self.grad_bias_input = np.zeros(self.nodes)
        self.grad_bias_hidden = np.zeros([self.layers, self.nodes])
        self.grad_bias_output = np.zeros(self.M_outputs)

        self.grad_weights_input = np.zeros([self.nodes, self.features])
        self.grad_weights_hidden = np.zeros([self.layers, self.nodes, self.nodes])
        self.grad_weights_output = np.zeros([self.M_outputs,self.nodes])

        self.activations = np.zeros([self.layers,self.nodes])
        self.bias = np.random.normal(size=[self.layers, self.nodes])
        self.bias_output = np.random.normal(size=self.M_outputs)

        self.output = np.zeros(self.M_outputs)

        self.weights_input = np.random.random(size=[self.nodes,self.features])
        self.weights_hidden = np.random.normal(size=[self.layers, self.nodes, self.nodes])
        self.weights_output = np.random.random(size=[self.M_outputs,self.nodes])

        #print(np.shape(self.weights))

        if problem_type == "classification":
            self.compute_error_output = lambda activation, y: self.error_output_classification(activation, y)
            self.compute_output = lambda z: self.softmax(z)

        if hidden_activation == "sigmoid":
            self.compute_hidden_act = lambda z: self.sigmoid(z)

        if hidden_activation == "ReLU":
            self.compute_hidden_act = lambda z: self.ReLU(z)


    def train(self):
        batches = self.N_points//self.batch_size
        total_indices = np.arange(self.N_points)

        bar = Bar("Epoch ", max = self.epochs)
        for epoch in range(self.epochs):
            bar.next()
            for b in range(batches):
                indices = np.random.choice(total_indices, size=self.batch_size, replace=False)
                self.X  = self.X_data[indices]
                self.y = self.y_data[indices]
                for i in range(self.batch_size):
                    self.feed_forward(self.X[i])
                    self.backpropagate(self.X[i], self.y[i])
                self.update_parameters()

        bar.finish()


    def feed_forward(self, x):
        z = (self.weights_input @ x) + self.bias[0]
        self.activations[0] = self.compute_hidden_act(z)
        for l in range(1,self.layers):
            z = self.weights_hidden[l] @ self.activations[l-1] + self.bias[l]
            self.activations[l] = self.compute_hidden_act(z)
        z = self.weights_output @ self.activations[-1] + self.bias_output
        self.output = self.compute_output(z)

    def backpropagate(self, x, y):
        #Compute error at top layer
        self.error_output = self.compute_error_output(self.output, y)

        #update bias
        self.grad_bias_output += self.error_output
        self.grad_weights_output += np.outer(self.error_output, self.activations[-1])

        s = self.activations[-1]* (1 - self.activations[-1])
        self.error_hidden[-1] = (self.weights_output.T @ self.error_output)*s
        self.grad_bias_hidden[-1] += self.error_hidden[-1]
        self.grad_weights_hidden[-1] += np.outer(self.error_hidden[-1], self.activations[-2])


        for l in range(self.layers-2, 0, -1):
            s = self.activations[l] * (1 - self.activations[l])
            self.error_hidden[l] = (self.weights_hidden[l+1].T @ self.error_hidden[l+1])*s
            self.grad_bias_hidden[l] += self.error_hidden[l]
            self.grad_weights_hidden[l] += np.outer(self.error_hidden[l], self.activations[l-1])

        s = self.activations[0] * (1 - self.activations[0])
        self.error_hidden[0] = (self.weights_hidden[1].T @ self.error_hidden[1])*s
        self.grad_bias_hidden[0] += self.error_hidden[0]
        self.grad_weights_input += np.outer(self.error_hidden[0], x)


    def update_parameters(self):
        scale = self.eta/self.batch_size
        self.grad_bias_hidden *= scale
        self.grad_bias_output *= scale
        self.grad_weights_input *= scale
        self.grad_weights_hidden *= scale
        self.grad_weights_output *= scale

        self.weights_input -= self.grad_weights_input

        for l in range(self.layers):
            self.bias[l] -= self.grad_bias_hidden[l]
            self.weights_hidden[l] -= self.grad_weights_hidden[l]


        self.bias_output -= self.grad_bias_output
        self.weights_output -= self.grad_weights_output

        self.grad_bias_hidden[:,:] = 0.
        self.grad_bias_output[:] = 0.
        self.grad_weights_input[:,:] = 0.
        self.grad_weights_hidden[:,:,:] = 0.
        self.grad_weights_output[:,:] = 0.


    def predict(self, x):
        self.feed_forward(x)
        return self.output


    def softmax(self, z):
        Z = np.sum(np.exp(z))
        return np.exp(z)/Z

    @staticmethod
    def sigmoid(z):
        return 1./(1+np.exp(-z))

    @staticmethod
    def ReLU(z):
        idx = np.where(z <= 0)
        z[idx] = 0
        return z

    @staticmethod
    def error_output_classification(activation, y):
        return activation - y

    @staticmethod
    def error_output_classification_l2(activation, y):
        return activation - y

    @staticmethod
    def error_output_regression(activation, y):
        return None

import numpy as np
from progress.bar import Bar
np.random.seed(1001)

class FFNN():

    def __init__(self, hidden_layers, nodes, X_data, y_data, N_outputs, epochs=10, batch_size=1, eta = 0.1, problem_type="classification", hidden_activation="sigmoid", Lambda=0, gamma=0):
        self.layers = hidden_layers
        self.nodes = nodes
        self.X_data = X_data
        self.y_data = y_data
        self.N_points, self.features = np.shape(X_data)
        self.N_outputs = N_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta                    # learning rate
        self.Lambda = Lambda              # regularization parameter
        self.gamma = gamma                # momentum parameter

        mean = 0; std = 0.0001



        self.activations_input = np.zeros(self.nodes)
        self.activations_hidden = np.zeros([self.layers,self.nodes])
        self.activations_output = np.zeros(self.N_outputs)

        self.weights_input = np.random.normal(mean, std, size=[self.nodes,self.features])
        self.weights_hidden = np.random.normal(mean, std, size=[self.layers, self.nodes, self.nodes])
        self.weights_output = np.random.normal(mean, std, size=[self.N_outputs,self.nodes])


        self.bias_input = np.random.normal(mean, std, size=self.nodes)
        self.bias_hidden = np.random.normal(mean, std, size=[self.layers, self.nodes])
        self.bias_output = np.random.normal(mean, std, size=self.N_outputs)

        self.grad_weights_input = np.zeros([self.nodes, self.features])
        self.grad_weights_hidden = np.zeros([self.layers, self.nodes, self.nodes])
        self.grad_weights_output = np.zeros([self.N_outputs,self.nodes])

        self.grad_bias_input = np.zeros(self.nodes)
        self.grad_bias_hidden = np.zeros([self.layers, self.nodes])
        self.grad_bias_output = np.zeros(self.N_outputs)

        self.error_input = np.zeros(self.nodes)
        self.error_hidden = np.zeros([self.layers,self.nodes])
        self.error_output = np.zeros(self.N_outputs)



        self.tmp_weights_input = np.zeros([self.nodes,self.features])
        self.tmp_weights_hidden = np.zeros([self.layers, self.nodes, self.nodes])
        self.tmp_weights_output = np.zeros([self.N_outputs,self.nodes])
        self.tmp_bias_input = np.zeros(self.nodes)
        self.tmp_bias_hidden = np.zeros([self.layers, self.nodes])
        self.tmp_bias_output = np.zeros(self.N_outputs)

        if problem_type == "classification":
            self.compute_output = lambda z: self.softmax(z)

        if problem_type == "regression":
            self.compute_output = lambda z: z


        if hidden_activation == "sigmoid":
            self.compute_act = lambda z: self.sigmoid(z)

        if hidden_activation == "ReLU":
            self.compute_act = lambda z: self.ReLU(z)
            self.weights_input *= 1e-3
            self.weights_hidden *= 1e-3
            self.weights_output *= 1e-3

        if hidden_activation == "LeakyReLU":
            self.compute_act = lambda z: self.LeakyReLU(z)
            self.weights_input *= 1e-3
            self.weights_hidden *= 1e-3
            self.weights_output *= 1e-3

        if hidden_activation == "ELU":
            self.compute_act = lambda z: self.ELU(z)
            self.weights_input *= 1e-3
            self.weights_hidden *= 1e-3
            self.weights_output *= 1e-3



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
                    self.feed_forward(self.X[i])

                    self.backpropagate(self.X[i], self.y[i])
                self.update_parameters()


        bar.finish()


    def feed_forward(self, x):
        #Compute activations in input layer
        z = (self.weights_input @ x) + self.bias_input
        self.activations_input = self.compute_act(z)

        #Compute activations in first hidden layer
        z = self.weights_hidden[0] @ self.activations_input + self.bias_hidden[0]
        self.activations_hidden[0] = self.compute_act(z)
        #print("NN activation_hidden = ", self.activations_hidden)
        #Compute activations in remaining hidden layers
        for l in range(1, self.layers):
            z = self.weights_hidden[l] @ self.activations_hidden[l-1] + self.bias_hidden[l]
            self.activations_hidden[l] = self.compute_act(z)
        z = self.weights_output @ self.activations_hidden[-1] + self.bias_output
        self.activations_output = self.compute_output(z)

    def backpropagate(self, x, y):
        #Compute error at top layer
        self.error_output = self.compute_error_output(self.activations_output, y)

        #update bias and weights
        self.grad_bias_output += self.error_output
        self.grad_weights_output += np.outer(self.error_output, self.activations_hidden[-1]) + self.Lambda*self.weights_output

        s = self.activations_hidden[-1]*(1 - self.activations_hidden[-1])
        self.error_hidden[-1] = (self.weights_output.T @ self.error_output)*s
        self.grad_bias_hidden[-1] += self.error_hidden[-1]

        if self.layers >= 2:
            self.grad_weights_hidden[-1] += np.outer(self.error_hidden[-1], self.activations_hidden[-2]) + self.Lambda*self.weights_hidden[-1]
            for l in range(self.layers-2, 0, -1):
                s = self.activations_hidden[l] * (1 - self.activations_hidden[l])
                self.error_hidden[l] = (self.weights_hidden[l+1].T @ self.error_hidden[l+1])*s
                self.grad_bias_hidden[l] += self.error_hidden[l]
                self.grad_weights_hidden[l] += np.outer(self.error_hidden[l], self.activations_hidden[l-1]) + self.Lambda*self.weights_hidden[l]


            s = self.activations_hidden[0] * (1. - self.activations_hidden[0])
            self.error_hidden[0] = (self.weights_hidden[1].T @ self.error_hidden[1])*s
            self.grad_bias_hidden[0] += self.error_hidden[0]
            self.grad_weights_hidden[0] += np.outer(self.error_hidden[0], self.activations_input) + self.Lambda*self.weights_hidden[0]

        s = self.activations_input * (1. - self.activations_input)
        self.error_input = (self.weights_hidden[0].T @ self.error_hidden[0])*s
        self.grad_bias_input += self.error_input
        self.grad_weights_input += np.outer(self.error_input, x)


    def update_parameters(self):
        scale = self.eta/self.batch_size
        self.grad_bias_hidden *= scale
        self.grad_bias_output *= scale
        self.grad_weights_input *= scale
        self.grad_weights_hidden *= scale
        self.grad_weights_output *= scale

        self.tmp_weights_input = (self.grad_weights_input + self.gamma*self.tmp_weights_input)
        self.weights_input -= self.tmp_weights_input

        for l in range(self.layers):
            self.tmp_bias_hidden[l] = (self.grad_bias_hidden[l] + self.gamma*self.tmp_bias_hidden[l])
            self.bias_hidden[l] -= self.tmp_bias_hidden[l]

            self.tmp_weights_hidden[l] = (self.grad_weights_hidden[l] + self.gamma*self.tmp_weights_hidden[l])
            self.weights_hidden[l] -= self.tmp_weights_hidden[l]

        self.tmp_bias_output = (self.grad_bias_output + self.gamma*self.tmp_bias_output)
        self.bias_output -= self.tmp_bias_output

        self.tmp_weights_output = (self.grad_weights_output + self.gamma*self.tmp_weights_output)
        self.weights_output -= self.tmp_weights_output

        self.grad_bias_hidden[:,:] = 0.
        self.grad_bias_output[:] = 0.
        self.grad_weights_input[:,:] = 0.
        self.grad_weights_hidden[:,:,:] = 0.
        self.grad_weights_output[:,:] = 0.

    def predict(self, x):
        self.feed_forward(x)
        return self.activations_output

    @staticmethod
    def softmax(z):
        Z = np.sum(np.exp(z))
        return np.exp(z)/Z

    @staticmethod
    def sigmoid(z):
        return 1./(1+np.exp(-z))

    @staticmethod
    def ReLU(z):
        idx = np.where(z < 0)
        z[idx] = 0
        return z

    @staticmethod
    def LeakyReLU(z):
        idx_below_zero = np.where(z <= 0)
        z[idx_below_zero] *= 0.1
        return z

    @staticmethod
    def ELU(z):
        idx_below_zero = np.where(z <= 0)
        idx_above_zero = np.where(z >= 0)
        z[idx_below_zero] = np.exp(z[idx_below_zero])-1
        return z


    @staticmethod
    def compute_error_output(activation, y):
        return activation - y

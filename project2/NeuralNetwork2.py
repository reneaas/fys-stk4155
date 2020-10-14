import numpy as np
from progress.bar import Bar
np.random.seed(1001)


class FFNN():
    """
    Feed forward Neural Network with multiple hidden layers.
    """

    def __init__(self, layers, nodes, X_data, y_data, M_outputs, epochs=10, batch_size=100, eta = 0.1, momentum = 0.9, problem_type="classification", hidden_activation="sigmoid", regularization=None):
        self.layers = layers
        self.nodes = nodes
        self.X_data = X_data
        self.y_data = y_data
        self.N_points, self.features = np.shape(X_data)
        self.M_outputs = M_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.momentum = momentum


        #self.output = np.zeros(self.M_outputs)

        #self.activations = np.zeros([self.layers,self.nodes])
        #self.bias = np.random.normal(size=[self.layers, self.nodes])
        #self.bias_output = np.random.normal(size=self.M_outputs)

        #weights_input = np.random.random(size=[self.nodes,self.features])
        #weights_hidden = np.random.normal(size=[self.layers, self.nodes, self.nodes])
        #weights_output = np.random.random(size=[self.M_outputs, self.nodes])
        #self.weights = np.array([weights_input, weights_hidden, weights_output], dtype="object")


        #Her er ideen om indeksering og lagring av data.

        #Keep track of number of elements in each layer.
        self.num_cols = np.zeros(self.layers, dtype="int")
        self.num_rows  = np.zeros(self.layers, dtype="int")
        self.r_weights = np.zeros(self.layers+1, dtype="int")
        self.r_nodes = np.zeros(self.layers+1, dtype="int")

        #Specify number of rows and columns of the weight matrices of each layer.

        #input layer
        self.num_rows[0] = self.nodes
        self.num_cols[0] = self.features

        #hidden layers
        self.num_rows[1:-1] = self.nodes
        self.num_cols[1:-1] = self.nodes

        #top layer
        self.num_rows[-1] = self.M_outputs
        self.num_cols[-1] = self.nodes

        #Compute cumulative array elements.

        for i in range(self.layers):
            self.r_weights[i+1] = self.r_weights[i] + self.num_cols[i]*self.num_rows[i]
            self.r_nodes[i+1] = self.r_nodes[i] + self.num_rows[i]

        #Create weight matrix and activation matrix.
        self.weights = np.random.normal(0, 1, size=self.r_weights[-1])
        self.bias = np.random.normal(0, 1, size = self.r_nodes[-1])
        self.activations = np.zeros(self.r_nodes[-1])
        self.z = np.zeros(self.r_nodes[-1])
        self.errors = np.zeros(self.r_nodes[-1])

        self.grad_bias = np.zeros(self.r_nodes[-1])
        self.grad_weights = np.zeros(self.r_weights[-1])

        if problem_type == "classification":
            if regularization == None:
                self.output_error = lambda activation, y: self.output_error_classification(activation, y)

            self.compute_output = lambda z: self.softmax(z)

        if hidden_activation == "sigmoid":
            self.compute_hidden_act = lambda z: self.sigmoid(z)

        if hidden_activation == "ReLU":
            self.compute_hidden_act = lambda z: self.ReLU(z)


    def train(self):
        batches = self.N_points // self.batch_size
        data_indices = np.arange(self.N_points)

        bar = Bar("Epoch ", max = self.epochs)
        for epoch in range(self.epochs):
            bar.next()
            for batch in range(batches):
                indices = np.random.choice(data_indices, size=self.batch_size, replace=False)
                self.X  = self.X_data[indices]
                self.y = self.y_data[indices]
                for i in range(self.batch_size):
                    self.feed_forward(self.X[i])
                    #print("max z = ", np.max(self.z))
                    self.backpropagate(self.X[i], self.y[i])
                self.update_parameters()


        bar.finish()

    def update_parameters(self):
        self.grad_bias *= self.eta/self.batch_size
        self.grad_weights *= self.eta/self.batch_size
        for l in range(self.layers):
            self.bias[self.r_nodes[l]:self.r_nodes[l+1]] -= self.grad_bias[self.r_nodes[l]:self.r_nodes[l+1]]
            self.weights[self.r_weights[l]:self.r_weights[l+1]] -= self.grad_weights[self.r_weights[l]:self.r_weights[l+1]]

        self.grad_bias[:] = 0.
        self.grad_weights[:] = 0.



    def backpropagate(self, x_data, y):
        #Compute error at top layer
        self.errors[self.r_nodes[-2]:self.r_nodes[-1]] = self.output_error(self.activations[self.r_nodes[-2]:self.r_nodes[-1]], y)

        #update bias
        current_grad_bias = self.errors[self.r_nodes[-2]:self.r_nodes[-1]]
        self.grad_bias[self.r_nodes[-2]:self.r_nodes[-1]] += current_grad_bias
        #update weights
        current_grad_weights = np.outer(self.errors[self.r_nodes[-2]:self.r_nodes[-1]], self.activations[self.r_nodes[-3]:self.r_nodes[-2]]).flat[:]
        self.grad_weights[self.r_weights[-2]:self.r_weights[-1]] += current_grad_weights

        #Hidden layers
        for l in range(self.layers-2, 0, -1):
            s = self.compute_hidden_act(self.z[self.r_nodes[l]:self.r_nodes[l+1]])
            self.errors[self.r_nodes[l]:self.r_nodes[l+1]] = (self.weights[self.r_weights[l+1]:self.r_weights[l+2]].reshape(self.num_cols[l+1], self.num_rows[l+1]) @ self.errors[self.r_nodes[l+1]:self.r_nodes[l+2]] ) * s*(1-s)

            #update bias
            current_grad_bias = self.errors[self.r_nodes[l]:self.r_nodes[l+1]]
            self.grad_bias[self.r_nodes[l]:self.r_nodes[l+1]] += current_grad_bias


            #update weights
            current_grad_weights = np.outer(self.errors[self.r_nodes[l]:self.r_nodes[l+1]], self.activations[self.r_nodes[l-1]:self.r_nodes[l]]).flat[:]
            self.grad_weights[self.r_weights[l]:self.r_weights[l+1]] += current_grad_weights



        #Bottom layer
        s = self.compute_hidden_act(self.z[self.r_nodes[0]:self.r_nodes[1]])
        self.errors[self.r_nodes[0]:self.r_nodes[1]] = (self.weights[self.r_weights[1]:self.r_weights[2]].reshape(self.num_cols[1], self.num_rows[1]) @ self.errors[self.r_nodes[1]:self.r_nodes[2]] ) * s*(1-s)

        #update bias
        current_grad_bias = self.errors[self.r_nodes[0]:self.r_nodes[1]]
        self.grad_bias[self.r_nodes[0]:self.r_nodes[1]] += current_grad_bias


        #update weights
        current_grad_weights = np.outer(self.errors[self.r_nodes[0]:self.r_nodes[1]], x_data).flat[:]
        self.grad_weights[self.r_weights[0]:self.r_weights[1]] += current_grad_weights


    def feed_forward(self, x_data):
        #Compute bottom layer activations
        z = self.weights[self.r_weights[0]:self.r_weights[1]].reshape(self.num_rows[0], self.num_cols[0]) @ x_data + self.bias[self.r_nodes[0]:self.r_nodes[1]]
        self.z[self.r_nodes[0]:self.r_nodes[1]] = z
        self.activations[self.r_nodes[0]:self.r_nodes[1]] = self.compute_hidden_act(z)


        #Compute hidden layer activations
        for l in range(1,self.layers-1):
            z = self.weights[self.r_weights[l]: self.r_weights[l+1]].reshape(self.num_rows[l], self.num_cols[l]) @ self.activations[self.r_nodes[l-1]:self.r_nodes[l]] + self.bias[self.r_nodes[l]:self.r_nodes[l+1]]
            self.z[self.r_nodes[l]:self.r_nodes[l+1]] = z
            self.activations[self.r_nodes[l]:self.r_nodes[l+1]] = self.compute_hidden_act(z)

        #Compute top layer activations
        z = self.weights[self.r_weights[-2]:self.r_weights[-1]].reshape(self.num_rows[-1], self.num_cols[-1]) @ self.activations[self.r_nodes[-3]:self.r_nodes[-2]] + self.bias[self.r_nodes[-2]:self.r_nodes[-1]]
        self.z[self.r_nodes[-2]:self.r_nodes[-1]] = z
        self.activations[self.r_nodes[-2]:self.r_nodes[-1]] = self.compute_output(z)

    def predict(self, x):
        self.feed_forward(x)
        y = self.activations[self.r_nodes[-2]:self.r_nodes[-1]]
        return y

    def softmax(self, z):
        Z = np.sum(np.exp(z))
        #self.output = z/Z
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
    def output_error_classification(activation, y):
        return activation - y

    @staticmethod
    def output_error_classification_l2(activation, y):
        return activation - y

    @staticmethod
    def output_error_regression(activation, y):
        return None

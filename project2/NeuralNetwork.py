import numpy as np


class FFNN():

    def __init__(self, layers, nodes, X_data, y_data, M_outputs, epochs=10, batch_size=100, problem_type="classification", hidden_activation="sigmoid"):
        self.layers = layers
        self.nodes = nodes
        self.X_data = X_data
        self.y_data = y_data
        self.N_points, self.features = np.shape(X_data)
        self.M_outputs = M_outputs
        self.epochs = epochs
        self.batch_size = batch_size

        self.activations = np.zeros([layers,nodes])
        self.bias = np.random.normal(size=[layers, nodes])
        self.bias_output = np.random.normal(size=M_outputs)

        self.output = np.zeros(M_outputs)

        weights_input = np.random.random(size=[nodes,self.features])
        weights_hidden = np.random.normal(size=[layers, nodes, nodes])
        weights_output = np.random.random(size=[M_outputs,nodes])
        #self.weights = np.array([weights_input, weights_hidden, weights_output], dtype="object")

        print(np.shape(self.weights))

        if problem_type == "classification":
            self.output_error = lambda activation, y: self.output_error_classification(activation, y)
            self.compute_output = lambda z: self.softmax(z)

        if hidden_activation == "sigmoid":
            self.compute_hidden_act = lambda z: self.sigmoid(z)

    def softmax(self, z):
        Z = np.sum(np.exp(z))
        self.output = z/Z

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    @staticmethod
    def output_error_classification(activation, y):
        return activation - y

    @staticmethod
    def output_error_regression(activation, y):
        return None



"""
batches = self.N_points//self.batch_size
total_indices = np.arange(self.N_points)
for epoch in range(epochs):
    for b in range(batches):
        indices = np.random.choice(total_indices, size=self.batch_size, replace=False)
        self.X  = self.X_data[indices]
        self.y = self.y_data[indices]
        for i in range(np.shape(self.X)[0]):
            z = self.weights_input @ self.X[i] + self.bias[0]
            self.activations[0] = self.compute_hidden_act(z)
            for l in range(1,self.layers):
                z = self.weights[l] @ self.activations[l-1] + self.bias[l]
                self.activations[l] = self.compute_hidden_act(z)
            z = self.weights_output @ self.activations[-1] + self.bias_output
            self.compute_output(z)
"""

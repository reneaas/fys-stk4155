import numpy as np
from progress.bar import Bar
np.random.seed(1001)

class FFNN:
    def __init__(self, layers, problem_type, hidden_activation):
        self.n_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.normal(0,1,size=y)  for y in layers[1:]]
        self.weights = [np.random.normal(0,1,size=(y,x))/np.sqrt(x) for x, y in zip(layers[:-1], layers[1:])]

        if problem_type == "classification":
            self.compute_act = lambda z: self.softmax(z)

        if hidden_activation == "sigmoid":
            self.compute_hidden_act = lambda z: self.sigmoid(z)

        if hidden_activation == "relu":
            self.compute_hidden_act = lambda z: self.ReLU(z)

        if hidden_activation == "leakyrelu":
            self.compute_hidden_act = lambda z: self.LeakyReLU(z)


    def predict(self, a):
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = self.compute_hidden_act(w @ a + b)
        a = self.compute_act(self.weights[-1] @ a + self.biases[-1])
        return a

    def fit(self, X_train, y_train, batch_sz, eta, lamb, epochs, gamma):
        Npoints, features = np.shape(X_train)
        batches = Npoints//batch_sz
        data_indices = np.arange(Npoints)
        bar = Bar("Epoch ", max = epochs)

        tmp_grad_b = [np.zeros(b.shape) for b in self.biases]
        tmp_grad_w = [np.zeros(w.shape) for w in self.weights]
        for epoch in range(epochs):
            bar.next()
            for batch in range(batches):
                indices = np.random.choice(data_indices, size=batch_sz, replace=True)
                x = X_train[indices]
                y = y_train[indices]

                self.grad_b = [np.zeros(b.shape) for b in self.biases]
                self.grad_w = [np.zeros(w.shape) for w in self.weights]
                for i in range(batch_sz):
                    grad_b, grad_w = self.backpropagate(x[i], y[i])
                    self.grad_b = [b + db for b, db in zip(self.grad_b, grad_b)]
                    self.grad_w = [w + dw for w, dw in zip(self.grad_w, grad_w)]

                tmp_grad_b = [db + gamma*tmp_db for db, tmp_db in zip(self.grad_b, tmp_grad_b)]
                tmp_grad_w = [dw + gamma*tmp_dw for dw, tmp_dw in zip(self.grad_w, tmp_grad_w)]
                self.biases = [b - (eta/batch_sz)*db for b, db in zip(self.biases, tmp_grad_b)]
                self.weights = [w*(1-eta*lamb/Npoints) - (eta/batch_sz)*dw for w, dw in zip(self.weights, tmp_grad_w)]
        bar.finish()



    def backpropagate(self, x, y):
        weights = self.weights
        biases = self.biases
        n_layers = self.n_layers

        grad_b = [np.zeros(b.shape) for b in biases]
        grad_w = [np.zeros(w.shape) for w in weights]
        a = x
        activations = [x]
        Z = []
        Z_append = Z.append
        a_append = activations.append
        #Feed forward
        for b, w in zip(biases[:-1], weights[:-1]):
            z = w @ a + b
            a = self.compute_hidden_act(z)
            Z_append(z)
            a_append(a)
        z = weights[-1] @ a + biases[-1]
        a = self.compute_act(z)
        a_append(a)
        Z_append(z)

        #Backward pass
        delta = self.cost_derivative(activations[-1], y)
        grad_b[-1] = delta
        grad_w[-1] = np.outer(delta, activations[-2])


        for l in range(2, n_layers):
            z = Z[-l]
            s = self.sigmoid(z)
            delta = (weights[-l+1].T @ delta)*s*(1-s)
            grad_b[-l] = delta
            grad_w[-l] = np.outer(delta, activations[-l-1])

        return grad_b, grad_w


    def cost_derivative(self, activations, y):
        return activations-y

    @staticmethod
    def sigmoid(z):
        return 1./(1.0+np.exp(-z))

    @staticmethod
    def softmax(z):
        a = np.exp(z)
        Z = np.sum(a)
        return a/Z

    @staticmethod
    def ReLU(z):
        return z*(z > 0)

    @staticmethod
    def LeakyReLU(z):
        return 0.1*z*(z <= 0) + z*(z > 0)

    def softmax_derivate(self, z):
        s = self.softmax(z)
        return s*(1-s)

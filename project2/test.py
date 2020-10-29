from NeuralNetwork import FFNN
import numpy as np
import matplotlib.pyplot as plt
from time import time
from functions import scale_data, mnist_data, test_model_mnist, design_matrix, read_data, split_data
import unittest

class Test_NN(unittest.TestCase):

    def test_function(self, x,y):
        hidden_layers = 1
        nodes = 2
        features = 2
        N_outputs = 1
        batch_size=1
        epochs=1
        eta=0.1
        problem_type="regression"
        hidden_activation="sigmoid"
        Lambda=0.
        gamma=0.
        tol = 10e-8

        X_data = np.array([[x,y]])
        y_data = np.array([x+y])

        my_solver = FFNN(hidden_layers=hidden_layers, nodes=nodes, X_data=X_data, y_data=y_data, N_outputs=N_outputs, epochs=epochs, batch_size=batch_size, eta=eta, problem_type=problem_type, hidden_activation=hidden_activation, Lambda=Lambda, gamma=gamma)

        mu = 0
        std = 0.1

        w_input = np.random.normal(mu, std, size=(2,2))
        b_input = np.random.normal(mu,std, size=2)

        w_hidden = np.random.normal(mu, std, size=(2,2))
        b_hidden = np.random.normal(mu,std, size=2)

        w_output = np.random.normal(mu,std, size=(1,2))
        b_output = np.random.normal(mu,std, size=1)

        my_solver.weights_input = np.copy(w_input)
        my_solver.bias_input = np.copy(b_input)

        my_solver.weights_hidden[0] = np.copy(w_hidden)
        my_solver.bias_hidden[0] = np.copy(b_hidden)

        my_solver.weights_output = np.copy(w_output)
        my_solver.bias_output = np.copy(b_output)

        """
        print("Input weights")
        print("weights_input = ", w_input)
        print("Weights input NN = ", my_solver.weights_input)

        print("Hidden layer weights")
        print("Weights hidden = ", w_hidden)
        print("Weights hidden NN = ", my_solver.weights_hidden)

        print("Output layer weights")
        print("Weights output = ", w_output)
        print("Weights output NN = ", my_solver.weights_output)


        print("Input bias")
        print("bias_input = ", b_input)
        print("bias input NN = ", my_solver.bias_input)

        print("Hidden layer bias")
        print("Weights hidden = ", b_hidden)
        print("bias hidden NN = ", my_solver.bias_hidden)

        print("Output layer bias")
        print("bias output = ", b_output)
        print("bias output NN = ", my_solver.bias_output)
        """

        X = np.array([x,y])


        #Feed forward
        activations_input = sigmoid(w_input@X + b_input)
        activations_hidden = sigmoid(w_hidden@activations_input + b_hidden)
        activations_output = w_output@activations_hidden + b_output

        print("Output activations = ", activations_output)
        my_solver.feed_forward(X)
        print("NN output activations = ", my_solver.activations_output)

        msg_eq = "Output activations =! NN output activations \n O_a = %f , NN_O_a = %f" % (activations_output, my_solver.activations_output)
        success_eq = (np.abs(activations_output-my_solver.activations_output) < tol)
        assert success_eq, msg_eq




        error_output = activations_output - y_data
        error_hidden = (w_output.T@error_output)*(activations_hidden*(1-activations_hidden))
        error_input = (w_hidden.T@error_hidden)*(activations_input*(1-activations_input))

        print("error_output = ", error_output)
        my_solver.backpropagate(X,y_data)
        print("NN error_output = ", my_solver.error_output)

        msg_eq = "Error output =! NN error output \n e = %f , NN_e = %f" % (error_output, my_solver.error_output)
        success_eq = (np.abs(error_output-my_solver.error_input_output) < tol)
        assert success_eq, msg_eq



if __name__ == '__main__':

    unittest.main()

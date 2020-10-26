from NeuralNetwork import FFNN
import numpy as np
from time import time
from functions import scale_data, mnist_data, test_model_mnist, design_matrix


#Set up data:
Ntrain = 10000
Ntest = 1000
X_train, Y_train, X_test, Y_test = mnist_data(Ntrain, Ntest)

my_solver = FFNN(layers = 5, nodes = 100, X_data = X_train, y_data = Y_train, N_outputs = 10, hidden_activation = "ReLU", epochs = 10, Lambda=0.0001, gamma=0.1)

start = time()
my_solver.train()
end = time()
timeused = end - start
print("Timeused = ", timeused)
accuracy = test_model_mnist(my_solver, X_test, Y_test, Ntest)
print("Accuracy = ", accuracy)

from NeuralNetwork import FFNN
import numpy as np
import matplotlib.pyplot as plt
from time import time
from functions import scale_data, mnist_data, results_model_mnist, design_matrix, read_data, split_data

np.random.seed(1)


Ntrain = 800
Ntest = 200
def train_and_test_mnist(Ntrain, Ntest, hidden_layers, nodes, N_outputs, hidden_activation, epochs, Lambda, gamma):
    X_train, Y_train, X_test, Y_test = mnist_data(Ntrain, Ntest)
    my_solver = FFNN(hidden_layers = hidden_layers, nodes = nodes, X_data = X_train, y_data = Y_train, N_outputs = N_outputs, hidden_activation = hidden_activation, epochs = epochs, Lambda=Lambda, gamma=gamma)
    start = time()
    my_solver.train()
    end = time()
    timeused = end - start
    print("Timeused = ", timeused)
    accuracy = results_model_mnist(my_solver, X_test, Y_test, Ntest)
    print("Accuracy = ", accuracy)
    return accuracy


#train_and_test_mnist(Ntrain=Ntrain, Ntest=Ntest, hidden_layers = 2, nodes = 10, N_outputs = 10, hidden_activation="sigmoid", epochs=100, Lambda = 0.0001, gamma = 0.)

def heat_map_mnist(start_nodes, end_nodes, start_layers, end_layers):
    nodes = np.linspace(start_nodes, end_nodes, end_nodes-start_nodes+1)
    layers = np.linspace(start_layers, end_layers, end_layers-start_layers+1)

    n, m = len(nodes), len(layers)
    accuracy = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            N_nodes = nodes[i]
            N_layers = layers[j]
            accuracy[i,j] = train_and_test_mnist(Ntrain=Ntrain, Ntest=Ntest, hidden_layers = int(N_layers), nodes = int(N_nodes), N_outputs = 10, hidden_activation="sigmoid", epochs=10, Lambda = 0.0001, gamma = 0.9)
            print("Nodes   Layers   Accuracy ")
            print(N_nodes, N_layers, accuracy[i,j])




    nodes, layers = np.meshgrid(nodes, layers)
    plt.contourf(nodes, layers, accuracy.T)
    plt.show()


#heat_map_mnist(start_nodes = 10, end_nodes = 50, start_layers = 2, end_layers = 10)


def regression_franke_func(hidden_layers, nodes, epochs, batch_size, eta, Lambda, gamma, degree):
    N = 1000
    sigma = 0.1
    filename = "datasets/frankefunction_dataset_N_{0}_sigma_{1}.txt".format(N,sigma)

    X_data, Y_data, z_data = read_data(filename)
    my_design_matrix, z_data = design_matrix(X_data, Y_data, z_data, degree)

    X_train, X_test, z_train, z_test = split_data(my_design_matrix, z_data, N, fraction_train = 0.8)
    n_train = len(X_train)

    problem_type = "regression"
    hidden_activation = "sigmoid"
    my_solver = FFNN(hidden_layers=hidden_layers, nodes=nodes, X_data=X_train, y_data=z_train, N_outputs=1, epochs=epochs, batch_size=batch_size, eta = eta, problem_type=problem_type, hidden_activation=hidden_activation, Lambda=Lambda, gamma=gamma)

    my_solver.train()
    test_result = np.zeros(N-n_train)
    for i in range(N-n_train):
        test_result[i] = my_solver.predict(X_test[i])

    MSE = np.mean((test_result - z_test)**2)
    print("MSE = ", MSE)
    return MSE

def results_franke_func():
    degs = np.array([i for i in range(1,20)])
    for i in range(len(degs)):
        MSE = regression_franke_func(hidden_layers=2, nodes = 5, epochs = 10, batch_size = 10, eta = 0.1, Lambda = 0., gamma = 0.9, degree = degs[i])

#results_franke_func()

def sigmoid(x):
    return 1./(1+np.exp(-x))

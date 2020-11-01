from NeuralNetwork2 import FFNN
from NN2 import FFNN2
import numpy as np
import matplotlib.pyplot as plt
from time import time
from functions import scale_data, mnist_data, results_model_mnist, design_matrix, read_data, split_data

np.random.seed(1)


Ntrain = 60000
Ntest = 10000
def train_and_test_mnist(Ntrain, Ntest, hidden_layers, nodes, N_outputs, hidden_activation, epochs, Lambda, gamma, eta):
    X_train, Y_train, X_test, Y_test = mnist_data(Ntrain, Ntest)
    my_solver = FFNN(hidden_layers = hidden_layers, nodes = nodes, X_data = X_train, y_data = Y_train, N_outputs = N_outputs, problem_type="classification", hidden_activation = hidden_activation, epochs = epochs, Lambda=Lambda, gamma=gamma, eta = eta)
    start = time()
    my_solver.train()
    end = time()
    timeused = end - start
    print("Timeused = ", timeused)
    accuracy = results_model_mnist(my_solver, X_test, Y_test, Ntest)
    print("Accuracy = ", accuracy)
    return accuracy

#print("Old Network: ")
train_and_test_mnist(Ntrain=Ntrain, Ntest=Ntest, hidden_layers = 1, nodes = 30, N_outputs = 10, hidden_activation="sigmoid", epochs=30, Lambda = 0., gamma = 0., eta = 0.5)

def train_and_test_mnist2(Ntrain, Ntest, hidden_layers, features, outputs, eta, epochs, nodes, batch_sz, lamb, gamma):
    X_train, Y_train, X_test, Y_test = mnist_data(Ntrain, Ntest)
    layers = [features] + [nodes]*hidden_layers + [outputs]
    #print(layers)
    print(layers)
    my_solver = FFNN2(layers, problem_type="classification", hidden_activation="sigmoid")
    #my_solver.fit_with_predict(X_train, Y_train, X_test, Y_test, batch_sz = batch_sz, eta=eta, lamb=lamb, epochs = epochs, gamma = gamma)
    start = time()
    my_solver.fit(X_train, Y_train, batch_sz = batch_sz, eta=eta, lamb=lamb, epochs = epochs, gamma = gamma)
    end = time()
    timeused = end - start
    print("Timeused = ", timeused)
    accuracy = test_model_mnist(my_solver, X_test, Y_test, Ntest)

#print("New network")
#train_and_test_mnist2(Ntrain=Ntrain, Ntest=Ntest, hidden_layers = 1, features=28*28, outputs=10, eta=0.5, epochs=10, nodes=30, batch_sz=10, lamb=0., gamma = 0.)


def grid_search_mnist_learningrate__lambda():
    learning_rates = np.array([10**(-i) for i in range(1,4)])
    Lambdas = np.array([10**(-i) for i in range(4, 9)])
    n, m = len(learning_rates), len(Lambdas)
    accuracy = np.zeros([n, m])
    print("Learning rate   Lambda   Accuracy ")
    for i in range(n):
        for j in range(m):
            accuracy[i,j] = train_and_test_mnist(Ntrain=Ntrain, Ntest=Ntest, hidden_layers = 1, nodes = 50, N_outputs = 10, hidden_activation="sigmoid", epochs=10, Lambda = Lambdas[j], gamma = 0.7, eta = learning_rates[i])
            print(learning_rates[i], Lambdas[j], accuracy[i,j])
    learning_rates, Lambdas = np.meshgrid(learning_rates, Lambdas)
    plt.contourf(learning_rates, Lambdas, accuracy.T, cmap="inferno", levels=40)
    plt.colorbar()
    plt.xlabel("Learning rate")
    plt.ylabel("Lambda")
    plt.savefig("grid_search_mnist_learningrate__lambda.pdf")
    plt.close()

#grid_search_mnist_learningrate__lambda()

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

    #MSE = np.mean((test_result - z_test)**2)
    R2 = 1 - np.sum((z_test - test_result) ** 2) / np.sum((z_test - np.mean(z_test)) ** 2)
    print("R2 = ", R2)
    return R2

def results_franke_func():
    degs = np.array([i for i in range(1,20)])
    for i in range(len(degs)):
        MSE = regression_franke_func(hidden_layers=1, nodes = 10, epochs = 100, batch_size = 10, eta = 0.1, Lambda = 1e-5, gamma = 0.7, degree = degs[i])

#test_franke_func()

def sigmoid(x):
    return 1./(1+np.exp(-x))

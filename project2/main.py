from NeuralNetwork import FFNN
import numpy as np
import matplotlib.pyplot as plt
from time import time
from functions import scale_data, mnist_data, test_model_mnist, design_matrix, read_data, split_data

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
    accuracy = test_model_mnist(my_solver, X_test, Y_test, Ntest)
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

def test_franke_func():
    degs = np.array([i for i in range(1,20)])
    for i in range(len(degs)):
        MSE = regression_franke_func(hidden_layers=2, nodes = 5, epochs = 10, batch_size = 10, eta = 0.1, Lambda = 0., gamma = 0.9, degree = degs[i])

test_franke_func()

def sigmoid(x):
    return 1./(1+np.exp(-x))

def test_function(x,y):
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

    #print("Output activations = ", activations_output)
    my_solver.feed_forward(X)
    #print("NN output activations = ", my_solver.activations_output)


    error_output = activations_output - y_data
    error_hidden = (w_output.T@error_output)*(activations_hidden*(1-activations_hidden))
    error_input = (w_hidden.T@error_hidden)*(activations_input*(1-activations_input))

    print("error_output = ", error_output)
    my_solver.backpropagate(X,y_data)
    print("NN error_output = ", my_solver.error_output)


#test_function(3,2)

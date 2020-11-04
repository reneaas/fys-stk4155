import os
import numpy as np

def grid_search_learningrate_num_points(hidden_layers = 1, nodes = 30, num_outputs = 10, epochs = 30, batch_size = 10, lamb = 0., gamma = 0., features = 28*28):
    learning_rates = [10**(-i) for i in range(5,0,-1)]
    data_size = [100, 1000, 10000, 30000, 60000]

    accuracy = np.zeros([len(learning_rates), len(data_size)])

    for i in range(len(learning_rates)):
        for j in range(len(data_size)):
            print("eta, num_points = ", learning_rates[i], data_size[j])
            outfilename = "_".join(["accuracy", str(learning_rates[i]), str(data_size[j])]) + ".txt"
            args = ["./main.out", str(hidden_layers), str(nodes), str(num_outputs), str(epochs), str(batch_size), str(learning_rates[i]), str(lamb), str(gamma), str(features), str(data_size[j]), outfilename]
            command = " ".join(args)
            os.system(command)

            with open(outfilename, "r") as infile:
                lines = infile.readlines()
                vals = lines[0].split()
                accuracy[i,j] = float(vals[-1])

            os.system("rm" + " " + outfilename)

    result_dir = "results/"
    outfilename = "grid_search_learningrate_numpoints.txt"

    with open(outfilename, "w") as outfile:
        for i in range(len(learning_rates)):
            for j in range(len(data_size)):
                eta = learning_rates[i]
                num_points = data_size[j]
                outfile.write(str(eta) + " " + str(num_points) + " " + str(accuracy[i,j]))
                outfile.write("\n")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    os.system("mv" + " " + outfilename + " " + result_dir)

#grid_search()


def grid_search_neurons_epochs(hidden_layers=1, learning_rate = 0.1, num_outputs = 10, batch_size = 10, lamb = 0., gamma = 0., features = 28*28, data_sz = 60000):
    neurons = [1, 10, 30, 50, 100]
    epochs = [1, 10, 30, 50, 100]


    accuracy = np.zeros([len(neurons), len(epochs)])

    for i in range(len(neurons)):
        for j in range(len(epochs)):
            num_neurons = neurons[i]
            num_epochs = epochs[j]
            print("nodes, epochs = ", num_neurons, num_epochs)
            outfilename = "_".join(["accuracy", str(num_neurons), str(num_epochs)]) + ".txt"
            args = ["./main.out", str(hidden_layers), str(num_neurons), str(num_outputs), str(num_epochs), str(batch_size), str(learning_rate), str(lamb), str(gamma), str(features), str(data_sz), outfilename]
            command = " ".join(args)
            os.system(command)

            with open(outfilename, "r") as infile:
                lines = infile.readlines()
                vals = lines[0].split()
                accuracy[i,j] = float(vals[-1])

            os.system("rm" + " " + outfilename)

    result_dir = "results/"
    outfilename = "grid_search_neurons_epochs.txt"

    with open(outfilename, "w") as outfile:
        for i in range(len(neurons)):
            for j in range(len(epochs)):
                num_neurons = neurons[i]
                num_epochs = epochs[j]
                outfile.write(str(num_neurons) + " " + str(num_epochs) + " " + str(accuracy[i,j]))
                outfile.write("\n")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    os.system("mv" + " " + outfilename + " " + result_dir)


def grid_search_lambda_momentum(hidden_layers=1, epochs = 10, learning_rate = 0.1, num_outputs = 10, batch_size = 10, features = 28*28, data_sz = 60000, neurons = 100):
    momentum = [0.5, 0.6, 0.7, 0.8, 0.9]
    lambs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


    accuracy = np.zeros([len(momentum), len(lambs)])

    for i in range(len(momentum)):
        for j in range(len(lambs)):
            gamma = momentum[i]
            lamb = lambs[j]
            print("momentum, lamb = ", gamma, lamb)
            outfilename = "_".join(["accuracy", str(gamma), str(lamb)]) + ".txt"
            args = ["./main.out", str(hidden_layers), str(neurons), str(num_outputs), str(epochs), str(batch_size), str(learning_rate), str(lamb), str(gamma), str(features), str(data_sz), outfilename]
            command = " ".join(args)
            os.system(command)

            with open(outfilename, "r") as infile:
                lines = infile.readlines()
                vals = lines[0].split()
                accuracy[i,j] = float(vals[-1])

            os.system("rm" + " " + outfilename)

    result_dir = "results/"
    outfilename = "grid_search_regularization_lambda.txt"

    with open(outfilename, "w") as outfile:
        for i in range(len(momentum)):
            for j in range(len(lambs)):
                gamma = momentum[i]
                lamb = lambs[j]
                outfile.write(str(gamma) + " " + str(lamb) + " " + str(accuracy[i,j]))
                outfile.write("\n")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    os.system("mv" + " " + outfilename + " " + result_dir)


def classification_test():
    hidden_layers = 1
    num_neurons = 30
    num_outputs = 10
    num_epochs=10
    batch_size=10
    learning_rate=0.1
    lamb=0.
    gamma=0.
    features=28*28
    data_sz=60000
    outfilename = "test.txt"
    args = ["./main.out", str(hidden_layers), str(num_neurons), str(num_outputs), str(num_epochs), str(batch_size), str(learning_rate), str(lamb), str(gamma), str(features), str(data_sz), outfilename]
    command = " ".join(args)
    os.system(command)
    os.system("rm test.txt")


def regression_test():
    hidden_layers = 1
    num_neurons = 10
    num_outputs = 1
    num_epochs=100
    batch_size=10
    learning_rate=0.1
    lamb=1e-5
    gamma=0.7
    features=28
    data_sz=10000
    outfilename = "test.txt"
    args = ["./main.out", str(hidden_layers), str(num_neurons), str(num_outputs), str(num_epochs), str(batch_size), str(learning_rate), str(lamb), str(gamma), str(features), str(data_sz), outfilename]
    command = " ".join(args)
    os.system(command)
    os.system("rm test.txt")


#grid_search_learningrate_num_points()
#grid_search_neurons_epochs()
#grid_search_lambda_momentum()
#classification_test()
regression_test()

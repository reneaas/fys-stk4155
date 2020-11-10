import numpy as np
import os

#args = ["g++-10", "-o", "main.out", "layer.cpp", "neural_network.cpp", "classification_main.cpp", "-larmadillo", "-O3"]
args = ["c++", "-o", "main.out", "layer.cpp", "neural_network.cpp", "classification_main.cpp", "-larmadillo", "-O3", "-std=c++11"]

command = " ".join(args)
os.system("echo " + command)
os.system(command) #Compile .cpp files


def single_run():
    hidden_layers = 1
    nodes = 100
    lamb = 1
    gamma = 0.5
    epochs = 10
    batch_sz = 10
    eta = 0.01
    hidden_act = "leaky_relu"

    outfilename = "test.txt"
    args = ["./main.out", str(hidden_layers), str(nodes), str(lamb), str(gamma), str(epochs), str(batch_sz), str(eta), outfilename, hidden_act]
    command = " ".join(args)
    os.system(command)
    os.system("cat test.txt")
    os.system("rm test.txt")

#single_run()

def grid_search():
    #num_nodes = [1, 10, 30, 50, 100]
    #num_epochs = [1, 10, 30, 50, 100]

    #num_nodes = [1, 10]
    #num_epochs = [1, 10]
    #x_len = len(num_nodes)
    #y_len = len(num_epochs)

    lambdas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    gammas = [0.5, 0.6, 0.7, 0.8, 0.9]

    #lambdas = [1e-2, 1e-3]
    #gammas = [0.5, 0.6]
    x_len = len(lambdas)
    y_len = len(gammas)

    accuracy_val = np.zeros([x_len, y_len])
    accuracy_test = np.zeros([x_len, y_len])

    hidden_layers = 1
    nodes = 100
    lamb = 0.
    gamma = 0.
    epochs = 30
    batch_sz = 10
    eta = 0.01
    hidden_act = "relu"


    for i in range(x_len):
        for j in range(y_len):
            lamb = lambdas[i]
            gamma = gammas[j]
            print("lamb = {}, gamma = {}".format(lamb, gamma))

            outfilename = "_".join(["accuracy", str(nodes), str(epochs)]) + ".txt"

            args = ["./main.out", str(hidden_layers), str(nodes), str(lamb), str(gamma), str(epochs), str(batch_sz), str(eta), outfilename, hidden_act]
            command = " ".join(args)
            os.system(command)

            with open(outfilename, "r") as infile:
                line = infile.readline()
                vals = line.split()
                accuracy_val[i,j] = float(vals[0])
                accuracy_test[i,j] = float(vals[1])

            os.system(" ".join(["rm", outfilename]))

    outfilename_val = "grid_search_lamb_gamma_leaky_relu_val.txt"
    with open(outfilename_val, "w") as outfile:
        outfile.write("lamb gamma accuracy_val\n")
        for i in range(x_len):
            for j in range(y_len):
                lamb = lambdas[i]
                gamma = gammas[j]

                args = [str(nodes), str(epochs), str(accuracy_val[i,j])]
                line = " ".join(args)
                outfile.write(line)
                outfile.write("\n")

    outfilename_test = "grid_search_lamb_gamma_leaky_relu_test.txt"
    with open(outfilename_test, "w") as outfile:
        outfile.write("nodes epochs accuracy_test\n")
        for i in range(x_len):
            for j in range(y_len):
                lamb = lambdas[i]
                gamma = gammas[j]

                #lamb = lambdas[i]
                #gamma = gammas[j]
                args = [str(lamb), str(gamma), str(accuracy_test[i,j])]
                line = " ".join(args)
                outfile.write(line)
                outfile.write("\n")
    os.system("mv *.txt results/classification/")
#grid_search()

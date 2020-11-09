import numpy as np
import os

args = ["g++-10", "-o", "main.out", "layer.cpp", "neural_network.cpp", "regression_main.cpp", "-larmadillo", "-O3"]
command = " ".join(args)
os.system("echo " + command)
os.system(command) #Compile .cpp files


def single_run():
    hidden_layers = 1
    nodes = 30
    lamb = 1e-4
    gamma = 0.5
    epochs = 100
    batch_sz = 100
    eta = 0.1
    hidden_act = "relu"
    deg = 5

    outfilename = "test.txt"
    args = ["./main.out", str(hidden_layers), str(nodes), str(lamb), str(gamma), str(epochs), str(batch_sz), str(eta), outfilename, hidden_act, str(deg)]
    command = " ".join(args)
    os.system(command)
    os.system("cat test.txt")
    os.system("rm test.txt")

#single_run()


def grid_search():
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    degrees = [d for d in range(1, 10)]
    #layers = [1,2]
    #degrees = [1, 2]
    #num_nodes = [1, 10]
    #degrees = [1, 2]
    x_len = len(layers)
    y_len = len(degrees)

    #lambdas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    #gammas = [0.5, 0.6, 0.7, 0.8, 0.9]
    #x_len = len(lambdas)
    #y_len = len(gammas)

    r2_val = np.zeros([x_len, y_len])
    r2_test = np.zeros([x_len, y_len])

    hidden_layers = 1
    nodes = 30
    lamb = 0.
    gamma = 0.
    epochs = 100
    batch_sz = 100
    eta = 0.1
    hidden_act = "relu"
    deg = 5


    for i in range(x_len):
        for j in range(y_len):

            #nodes = num_nodes[i]
            #deg = degrees[j]
            print("layers = {}, deg = {}".format(hidden_layers, deg))

            outfilename = "_".join(["r2", str(hidden_layers), str(deg)]) + ".txt"

            args = ["./main.out", str(hidden_layers), str(nodes), str(lamb), str(gamma), str(epochs), str(batch_sz), str(eta), outfilename]
            args = ["./main.out", str(hidden_layers), str(nodes), str(lamb), str(gamma), str(epochs), str(batch_sz), str(eta), outfilename, hidden_act, str(deg)]
            command = " ".join(args)
            os.system(command)

            with open(outfilename, "r") as infile:
                line = infile.readline()
                vals = line.split()
                r2_val[i,j] = float(vals[0])
                r2_test[i,j] = float(vals[1])

            os.system(" ".join(["rm", outfilename]))

    outfilename_val = "regression_grid_search_layers_deg_val.txt"
    with open(outfilename_val, "w") as outfile:
        outfile.write("layers degree r2_val\n")
        for i in range(x_len):
            for j in range(y_len):
                hidden_layers = layers[i]
                deg = degrees[j]

                args = [str(hidden_layers), str(deg), str(r2_val[i,j])]
                line = " ".join(args)
                outfile.write(line)
                outfile.write("\n")

    os.system("mv" + " " + outfilename_val + " " + "./results/regression/")

    outfilename_test = "regression_grid_search_layers_deg_test.txt"
    with open(outfilename_test, "w") as outfile:
        outfile.write("layers degree r2_test\n")
        for i in range(x_len):
            for j in range(y_len):
                hidden_layers = layers[i]
                deg = degrees[j]

                args = [str(hidden_layers), str(deg), str(r2_test[i,j])]
                line = " ".join(args)
                outfile.write(line)
                outfile.write("\n")

    os.system("mv" + " " + outfilename_test +  " " + "./results/regression/")



#grid_search()

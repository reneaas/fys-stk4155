import os
import numpy as np

def grid_search(hidden_layers = 1, nodes = 30, num_outputs = 10, epochs = 30, batch_size = 10, lamb = 0., gamma = 0., features = 28*28):
    learning_rates = [10**(-i) for i in range(1, 6)]
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
    outfilename = "grid_search_result.txt"

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

grid_search()

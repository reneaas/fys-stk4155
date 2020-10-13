import numpy as np



class FFNN():

    def __init__(self, layers, nodes, N_points, features, M_outputs, problem_type):
        self.layers = layers
        self.nodes = nodes
        self.N_points = N_points
        self.features = features
        self.M_outputs = M_outputs

        self.activations = np.zeros([layers,nodes])
        self.bias = np.random.normal(size=[layers, nodes])
        self.weights = np.random.normal(size=[layers, nodes, nodes])
        self.output = np.zeros(M_outputs)

        self.weights_input = np.random.random(size=[nodes,features])
        self.weights_output = np.random.random(size=[M_outputs,nodes])

        if problem_type == "classification":
            pass

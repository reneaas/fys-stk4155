from NeuralNetwork import FFNN
import numpy as np

X = np.array([[10,1],[2,3]]).reshape(2,2)

my_solver = FFNN(1, 1, X , 1, 1)

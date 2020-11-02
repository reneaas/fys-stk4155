
from NeuralNetwork import FFNN
import numpy as np
import matplotlib.pyplot as plt
from time import time
import unittest
from progress.bar import Bar
from main import sigmoid

class Test_NN(unittest.TestCase):

    def test_function(self):
        msg_eq = "some error message"
        success_eq = True
        assert success_eq, msg_eq

        msg_eq = "another error message"
        success_eq = True
        assert success_eq, msg_eq


if __name__ == '__main__':
    unittest.main()

from regression import Regression
import numpy as np
import matplotlib.pyplot as plt
import os

class Ridge(Regression):
    def __init__(self, Lambda):
        super().__init__()
        self.Lambda = Lambda

    def train(self):
        """
        Perform Ridge to find the parameters of the model denoted w.
        """
        A = self.X_train.T @ self.X_train
        shape = np.shape(A)
        A += self.Lambda*np.eye(shape[0])
        b = self.X_train.T @ self.f_train
        self.w = np.linalg.solve(A, b)

    def call_bootstrap(self, B):
        Lambdas = [1/10**i for i in range(-10,10)]
        for l in Lambdas:
            self.Lambda = l
            print("lambda = ", l)
            self.bootstrap(B)

    def call_cross_validate(self, k):
        Lambdas = [1/10**i for i in range(-10,10)]
        for l in Lambdas:
            self.Lambda = l
            print("lambda = ", l)
            self.k_fold_cross_validation(k)

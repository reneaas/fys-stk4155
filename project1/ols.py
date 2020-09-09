from regression import Regression
import numpy as np
np.random.seed(1001)

class OLS(Regression):
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train):
        """
        Perform ordinary least squares to find the parameters of the model denoted w.
        """
        A = X_train.T @ X_train
        b = X_train.T @ y_train
        self.w = np.linalg.solve(A, b)

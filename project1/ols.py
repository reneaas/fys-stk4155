from regression import Regression
import numpy as np
np.random.seed(1001)

class OLS(Regression):
    def __init__(self):
        super().__init__()

    def train(self):
        """
        Perform ordinary least squares to find the parameters of the model denoted w.
        """
        A = self.X_train.T @ self.X_train
        b = self.X_train.T @ self.f_train

        self.w = np.linalg.solve(A, b)

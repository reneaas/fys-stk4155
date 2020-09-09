from ols import OLS
import numpy as np

class Ridge(OLS):
    def __init__(self, Lambda):
        super().__init__()
        self.Lambda = Lambda

    def train(self, X_train, y_train):
        """
        Perform Ridge to find the parameters of the model denoted w.
        """
        A = X_train.T @ X_train
        shape = np.shape(A)
        A += self.Lambda*np.eye(shape[0])
        b = X_train.T @ y_train
        self.w = np.linalg.solve(A, b)

    def extract_MSE(self):
        return self.MSE_train, self.MSE_test

    def extract_R2(self):
        return self.R2_train, self.R2_test

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
        b = self.X_train.T @ self.y_train
        self.w = np.linalg.solve(A, b)

    def predict(self):
        """
        Computes predictions on the training and test dataset.
        It computes associated R2 scores on both traning and test data
        """
        self.y_train_predictions = self.X_train @ self.w
        self.y_test_predictions = self.X_test @ self.w

        self.R2_train = self.compute_R2_score(self.y_train, self.y_train_predictions)
        self.R2_test = self.compute_R2_score(self.y_test, self.y_test_predictions)
        self.MSE_train = self.compute_MSE(self.y_train, self.y_train_predictions)
        self.MSE_test = self.compute_MSE(self.y_test, self.y_test_predictions)
        #print("Training R2 score = ", R2_score_train)
        #print("Test R2 score = ", R2_score_test)
        #print("Weights = ", self.w)


    def extract_MSE(self):
        return self.MSE_train, self.MSE_test

    def extract_R2(self):
        return self.R2_train, self.R2_test

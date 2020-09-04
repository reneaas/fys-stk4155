from regression import Regression
import numpy as np

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
        R2_score_train = self.compute_R2_score(self.y_train, self.y_train_predictions)
        R2_score_test = self.compute_R2_score(self.y_test, self.y_test_predictions)
        print("Training R2 score = ", R2_score_train)
        print("Test R2 score = ", R2_score_test)
        print("Weights = ", self.w)

    def bootstrap_analysis(self, B):
        self.B = B #Number of resamples
        self.bootstrap_sampling()
        self.train()
        self.compute_statistics()

    def compute_statistics(self):
        self.w_mean = np.zeros(self.p)
        self.w_std = np.zeros(self.p)
        for i in range(self.p):
            self.w_mean[i] = np.mean(self.w_boots[:, i])
            self.w_std[i] = np.std(self.w_boots[:,i])

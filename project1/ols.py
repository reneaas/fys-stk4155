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
        R2_score_train = self.compute_R2_score(self.y_train, self.y_train_predictions)
        R2_score_test = self.compute_R2_score(self.y_test, self.y_test_predictions)
        print("Training R2 score = ", R2_score_train)
        print("Test R2 score = ", R2_score_test)
        print("Weights = ", self.w)

    def bootstrap_train(self, B):
        X_train_old = np.copy(self.X_train)
        y_train_old = np.copy(self.y_train)

        self.w_boots = np.zeros((B, self.p))
        for i in range(B):
            idx = np.random.randint(0,self.n_train, size=self.n_train)
            self.X_train = X_train_old[idx,:]
            self.y_train = y_train_old[idx]
            self.train()
            self.w_boots[i, :] = self.w[:]
        self.compute_statistics()

        self.X_train[:] = X_train_old[:]
        self.y_train[:] = y_train_old[:]

    def bootstrap_test(self, B):
        X_test_old = np.copy(self.X_test)
        y_test_old = np.copy(self.y_test)

        self.w_boots = np.zeros((B, self.p))
        for i in range(B):
            idx = np.random.randint(0,self.n_test, size=self.n_test)
            self.X_test = X_test_old[idx,:]
            self.y_test = y_test_old[idx]
            self.train()
            self.w_boots[i, :] = self.w[:]
        self.compute_statistics()

        self.X_train[:] = X_train_old[:]
        self.y_train[:] = y_train_old[:]

    def compute_statistics(self):
        self.w_mean = np.zeros(self.p)
        self.w_std = np.zeros(self.p)
        for i in range(self.p):
            self.w_mean[i] = np.mean(self.w_boots[:, i])
            self.w_std[i] = np.std(self.w_boots[:,i])
        print("w_mean = ", self.w_mean)
        print("w_std = ", self.w_std)

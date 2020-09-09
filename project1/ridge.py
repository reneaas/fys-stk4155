from ols import OLS
import numpy as np
import matplotlib.pyplot as plt
import os

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


    def bootstrap_ridge(self, B):
        Lambdas = [1/10**i for i in range(-10,10)]
        for l in Lambdas:
            self.Lambda = l
            print("lambda = ", l)
            self.bootstrap(B)

    def cross_validate_ridge(self, k):
        Lambdas = [1/10**i for i in range(-10,10)]
        for l in Lambdas:
            self.Lambda = l
            print("lambda = ", l)
            self.k_fold_cross_validation(k)

    def plot_regularization_path(self, filename_plots, path_plots):
        Lambdas = [1/10**i for i in range(-10,10)]
        R2_scores_train = []
        MSE_scores_train = []
        R2_scores_test = []
        MSE_scores_test = []
        for l in Lambdas:
            self.Lambda = l
            self.train(self.X_train, self.y_train)
            R2_train, MSE_train = self.predict(self.X_train, self.y_train)
            R2_test, MSE_test = self.predict(self.X_test, self.y_test)
            R2_scores_train.append(R2_train)
            MSE_scores_train.append(MSE_train)
            R2_scores_test.append(R2_test)
            MSE_scores_test.append(MSE_test)

        plt.plot(np.log10(Lambdas), R2_scores_train, label = "Training")
        plt.plot(np.log10(Lambdas), R2_scores_test, label = "Test")
        plt.xlabel(r"$\log_{10}( \lambda) $")
        plt.ylabel(r"$R^2$")
        plt.legend()
        plt.savefig(filename_plots[0])
        plt.close()

        plt.plot(np.log10(Lambdas), MSE_scores_train, label = "Training")
        plt.plot(np.log10(Lambdas), MSE_scores_test, label = "Test")
        plt.xlabel(r"$\log_{10}( \lambda) $")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(filename_plots[1])
        plt.close()

        if not os.path.exists(path_plots):
            os.makedirs(path_plots)

        filenames = " ".join(filename_plots)
        os.system(" ".join(["mv", filenames, path_plots]))

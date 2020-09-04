import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Regression:
    def __init__(self):
        """
        For now this is an empty useless init function
        """
        None

    def ReadData(self,filename, deg):
        """
        deg: Degree of the polynomial p(x,y)

        -------------------------------------------------
        Extracts:
        x, y: n data tuples (x,y).
        func_vals: n function values f(x,y)
        self.n: Number of datapoints n
        self.p: Number of features of the model
        """
        x = []
        y = []
        func_vals = []

        with open(filename, "r") as infile:
            lines = infile.readlines()
            self.n = len(lines)
            for line in lines:
                words = line.split()
                x.append(float(words[0]))
                y.append(float(words[1]))
                func_vals.append(float(words[2]))

        self.p = int((deg+1) * (deg+2) / 2) #Closed form expression for the number of features

        #Need to find a way to scale data (as suggested by the project text), but for now this works as it should.
        x = np.array(x)
        y = np.array(y)
        self.func_vals = np.array(func_vals)

        #Set up the design matrix
        self.design_matrix = np.zeros([self.n, self.p])
        self.design_matrix[:,0] = 1.0 #First column is simply 1s.
        col_idx = 1
        max_degree = 0
        # Sorry Kææsp, we had to change this part a little bit.... :)))))))
        while col_idx < self.p:
            max_degree += 1
            for i in range(max_degree+1):
                self.design_matrix[:,col_idx] = x[:]**(max_degree-i)*y[:]**i
                #print(col_idx,max_degree-i, i) #Nice way to visualize how the features are placed in the design matrix.
                col_idx += 1

    def SplitData(self):
        """
        Reshuffles and splits the data into a training set
        Training/test is by default 80/20 ratio.
        """
        #Reshuffle data to minimize risk of human bias
        shuffled_indices = np.random.permutation(self.n)
        self.design_matrix = self.design_matrix[shuffled_indices,:]
        self.func_vals = self.func_vals[shuffled_indices]

        #Split data into training and test set.
        self.train_n = 4*(self.n // 5) + 4*(self.n % 5)
        self.test_n = (self.n // 5) + (self.n % 5)
        self.X_train = self.design_matrix[:self.train_n,:]
        self.X_test = self.design_matrix[self.train_n:,:]
        self.y_train = self.func_vals[:self.train_n]
        self.y_test = self.func_vals[self.train_n:]

    def OLS(self):
        """
        Perform ordinary least squares to find the parameters of the model denoted w.
        """
        A = self.X_train.T @ self.X_train
        b = self.X_train.T @ self.y_train
        self.w = np.linalg.solve(A, b)

    def Predict(self):
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

    def compute_MSE(self):
        self.mean_square_error = np.mean((self.y_predictions - self.y_test)**2)
        print("Out-of-sample error = ", self.mean_square_error)

    def compute_R2_score(self,y_data, y_model):
        return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

    def kfold_CrossValidation(self, k):
        """
        Perform k-fold cross validation on the test set
        and produces a vector of R2 scores and an arithmetic mean R2 score.
        """
        cross_validation_size = self.test_n // k + self.test_n % k
        print(cross_validation_size)
        print(self.test_n)
        self.y_test_predictions = self.X_test @ self.w
        r2_scores = np.zeros(k)
        for i in range(k):
            y_data = self.y_test[i*cross_validation_size:(i+1)*cross_validation_size+1]
            y_model = self.y_test_predictions[i*cross_validation_size:(i+1)*cross_validation_size+1]
            r2_scores[i] = self.compute_R2_score(y_data, y_model)
        print(r2_scores)
        print(np.mean(r2_scores))

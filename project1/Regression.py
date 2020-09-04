import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Regression:
    def __init__(self, n):
        self.n = int(n)


    def ReadData(self,filename, deg): #p = number of features in you polynomial
        #Lese
        #Produsere Design matrise
        x = []
        y = []
        func_vals = []

        with open(filename, "r") as infile:
            lines = infile.readlines()
            for line in lines:
                words = line.split()
                x.append(float(words[0]))
                y.append(float(words[1]))
                func_vals.append(float(words[2]))

        self.p = int((deg+1) * (deg+2) / 2 - 1)

        x = np.array(x)
        y = np.array(y)
        self.func_vals = np.array(func_vals)


        # Scaling data
        x = x - np.mean(x)
        y = y - np.mean(y)
        self.func_vals = self.func_vals - np.mean(func_vals)

        #Set up the design matrix
        self.design_matrix = np.zeros([self.n, self.p])
        col_idx = 0
        max_degree = 0
        # Sorry Kææsp, we had to change this part a little bit.... :)))))))
        while col_idx < self.p:
            max_degree += 1
            for i in range(max_degree+1):
                self.design_matrix[:,col_idx] = x[:]**(max_degree-i)*y[:]**i
                col_idx += 1

    def SplitData(self):
        self.train_n = 4*(self.n // 5) + 4*(self.n % 5)
        self.test_n = (self.n // 5) + (self.n % 5)
        shuffled_indices = np.random.permutation(self.n)
        self.design_matrix = self.design_matrix[shuffled_indices,:]
        self.X_train = self.design_matrix[:self.train_n,:]
        self.X_test = self.design_matrix[self.train_n:,:]
        self.y_train = self.func_vals[:self.train_n]
        self.y_test = self.func_vals[self.train_n:]

    def OLS(self):
        self.A = self.X_train.T @ self.X_train
        self.b = self.X_train.T @ self.y_train
        self.w = np.linalg.solve(self.A, self.b)
        self.mse_train = np.mean( (self.X_train @ self.w - self.y_train)**2 )
        print("In-sample error = ", self.mse_train)
        print(self.w)


    def Predict(self):
        self.y_predictions = self.X_test @ self.w

    def MSE(self):
        self.mean_square_error = np.mean((self.y_predictions - self.y_test)**2)
        print("Out-of-sample error = ", self.mean_square_error)

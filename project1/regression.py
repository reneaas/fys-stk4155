import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1001)
from sklearn.model_selection import train_test_split
#import random


class Regression:
    def __init__(self):
        """
        For now this is an empty useless init function
        """

    def read_data(self,filename, deg):
        """
        deg: Degree of the polynomial p(x,y)

        -------------------------------------------------
        Extracts:
        x, y: n data tuples (x,y).
        func_vals: n function values f(x,y)
        self.n: Number of datapoints n
        self.p: Number of features of the model
        """

        self.x = []
        self.y = []
        self.func_vals = []

        with open(filename, "r") as infile:
            lines = infile.readlines()
            self.n = len(lines)
            for line in lines:
                words = line.split()
                self.x.append(float(words[0]))
                self.y.append(float(words[1]))
                self.func_vals.append(float(words[2]))

        self.p = int((deg+1) * (deg+2) / 2) #Closed form expression for the number of features

        #Need to find a way to scale data (as suggested by the project text), but for now this works as it should.
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.func_vals = np.array(self.func_vals)
        #self.scale_data()
        self.create_design_matrix() #Set up initial design matrix

    def scale_data(self):
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)
        self.x_std = np.std(self.x)
        self.y_std = np.std(self.y)
        self.func_vals_mean = np.mean(self.func_vals)
        self.func_vals_std = np.std(self.func_vals)

        self.x = (self.x - self.x_mean)/self.x_std
        self.y = (self.y - self.y_mean)/self.y_std
        self.func_vals = (self.func_vals - self.func_vals_mean)/self.func_vals_std


        #Set up the design matrix
    def create_design_matrix(self):
        self.design_matrix = np.zeros([self.n, self.p])
        self.design_matrix[:,0] = 1.0 #First column is simply 1s.
        col_idx = 1
        max_degree = 0
        while col_idx < self.p:
            max_degree += 1
            for i in range(max_degree+1):
                self.design_matrix[:,col_idx] = self.x[:]**(max_degree-i)*self.y[:]**i
                #print(col_idx,max_degree-i, i) #Nice way to visualize how the features are placed in the design matrix.
                col_idx += 1

    def split_data(self):
        """
        Reshuffles and splits the data into a training set
        Training/test is by default 80/20 ratio.
        """
        #Reshuffle data to minimize risk of human bias
        shuffled_idx = np.random.permutation(self.n)
        self.design_matrix = self.design_matrix[shuffled_idx,:]
        self.func_vals = self.func_vals[shuffled_idx]

        #Split data into training and test set.
        self.n_train = 4*(self.n // 5) + 4*(self.n % 5)
        self.n_test = (self.n // 5) + (self.n % 5)
        self.X_train = self.design_matrix[:self.n_train,:]
        self.X_test = self.design_matrix[self.n_train:,:]
        self.y_train = self.func_vals[:self.n_train]
        self.y_test = self.func_vals[self.n_train:]


    def compute_MSE(self, y_data, y_model):
        return np.mean((y_model - y_data)**2)


    def compute_R2_score(self,y_data, y_model):
        return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

    def confidence_intervals(self, sigma):
        """
        Compute 95% confidence intervals for each parameter w_i
        """
        self.confidence_interval = np.zeros([self.p,2])
        inv_X = np.linalg.inv(self.X_train.T@self.X_train)
        for i in range(self.p):
            standard_error = np.sqrt(sigma**2 * inv_X[i,i])
            lower_limit = self.w[i] - 1.96*standard_error
            upper_limit = self.w[i] + 1.96*standard_error
            self.confidence_interval[i,0] = lower_limit
            self.confidence_interval[i,1] = upper_limit


    def train(self, X_train, y_train):
        return None

    def bootstrap(self, B):
        self.w_boots = np.zeros((B, self.p))
        for i in range(B):
            idx = np.random.randint(0,self.n_train, size=self.n_train)
            X_train = self.X_train[idx,:]
            y_train = self.y_train[idx]
            self.train(X_train, y_train)
            self.w_boots[i, :] = self.w[:]
        self.compute_statistics(self.w_boots)

    def k_fold_cross_validation(self,k):

        self.w_k_fold = np.zeros((k, self.p))

        int_size = self.n_train//k
        rest = self.n_train%k

        fold_size = np.zeros(k, dtype ="int")
        for i in range(k):
            fold_size[i] = int_size + (rest > 0)
            rest -= 1


        row_ptr = np.zeros(k+1, dtype="int")
        for i in range(1,k+1):
            row_ptr[i] += row_ptr[i-1]+fold_size[i-1]

        for j in range(k):
            X_test = self.X_train[[i for i in range(row_ptr[j],row_ptr[j+1])], :]
            y_test = self.y_train[[i for i in range(row_ptr[j],row_ptr[j+1])]]
            idx = []
            for l in range(j):
                idx += [i for i in range(row_ptr[l],row_ptr[l+1])]
            for l in range(j+1,k):
                idx += [i for i in range(row_ptr[l],row_ptr[l+1])]
            X_train = self.X_train[idx, :]
            y_train = self.y_train[idx]
            self.train(X_train, y_train)
            self.w_k_fold[j, :] = self.w[:]

        self.compute_statistics(self.w_k_fold)


    def compute_statistics(self,w):
        self.w_mean = np.zeros(self.p)
        self.w_std = np.zeros(self.p)
        for i in range(self.p):
            self.w_mean[i] = np.mean(w[:, i])
            self.w_std[i] = np.std(w[:,i])
        self.w[:] = self.w_mean[:]
        #print("w_mean = ", self.w_mean)
        #print("w_std = ", self.w_std)

    def predict(self, X_data, y_data):
        y_prediction = X_data @ self.w
        R2_score = self.compute_R2_score(y_data, y_prediction)
        MSE = self.compute_MSE(y_data, y_prediction)
        return R2_score, MSE

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
        self.x = []
        self.y = []
        self.func_vals = []

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
        #self.x = []
        #self.y = []
        #self.func_vals = []

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

        self.x = (self.x - self.x_mean)/self.x_std
        self.y = (self.y - self.y_mean)/self.y_std


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


    def compute_MSE(self):
        self.mean_square_error = np.mean((self.y_predictions - self.y_test)**2)
        print("Out-of-sample error = ", self.mean_square_error)

    def compute_R2_score(self,y_data, y_model):
        return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

    def confidence_intervals(self, sigma):
        """
        Compute 95% confidence intervals for each parameter w_i
        """
        self.confidence_interval = np.zeros([self.p,2])
        standard_error = sigma/np.sqrt(self.n)  # Litt usikker på det her
        for i in range(self.p):
            lower_limit = self.w[i] - 1.96*standard_error
            upper_limit = self.w[i] + 1.96*standard_error
            self.confidence_interval[i,0] = lower_limit
            self.confidence_interval[i,1] = upper_limit

import numpy as np
from progress.bar import Bar
np.random.seed(1001)


class SGD_Regression():

    def __init__(self, Lambda=0):
        self.Lambda = Lambda

    def read_data(self,filename):
        """
        Reads data from a file where each line is of the form "x y z(x,y)".

        Input parameters:
        filename - filename of the file containing the dataset.
        """

        self.x = []
        self.y = []
        self.f_data = []


        with open(filename, "r") as infile:
            lines = infile.readlines()
            self.n = len(lines)
            for line in lines:
                words = line.split()
                self.x.append(float(words[0]))
                self.y.append(float(words[1]))
                self.f_data.append(float(words[2]))

        #Need to find a way to scale data (as suggested by the project text), but for now this works as it should.
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.f_data = np.array(self.f_data)
        self.scale_data()

        #Reshuffle data to minimize risk of human bias
        self.shuffled_idx = np.random.permutation(self.n)
        self.f_data = self.f_data[self.shuffled_idx] #Shuffle z = f(x,y) exactly once.

    def scale_data(self):
        """
        Scales the dataset according to standard score.
        """
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)
        self.x_std = np.std(self.x)
        self.y_std = np.std(self.y)
        self.f_data_mean = np.mean(self.f_data)
        self.f_data_std = np.std(self.f_data)

        self.x = (self.x - self.x_mean)/self.x_std
        self.y = (self.y - self.y_mean)/self.y_std
        self.f_data = (self.f_data - self.f_data_mean)/self.f_data_std


        #Set up the design matrix
    def create_design_matrix(self, deg):
        """
        Creates the design matrix of the model.

        Input parameters:
        deg - maximum degree of polynomial to be fitted
        """
        self.deg = deg
        self.p = int((deg+1) * (deg+2) / 2) #Closed-form expression for the number of features
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
        self.design_matrix = self.design_matrix[self.shuffled_idx,:] #Shuffle the design matrix

    def split_data(self, fraction = 0.2):
        """
        Splits the data into a training set and a test set
        Training/validate/test is by default 60/20/20 ratio.

        Input parameters:

        fraction_train - fraction of the dataset used for training.
        """

        fraction_train = 1-fraction*2

        #Split data into training and test set.
        self.n_train = int(fraction_train*self.n)
        self.n_validate = int(fraction*self.n)
        self.n_test = self.n - self.n_train - self.n_validate

        self.X_train = self.design_matrix[:self.n_train,:]
        self.X_validate = self.design_matrix[self.n_train:self.n_train + self.n_validate,:]
        self.X_test = self.design_matrix[self.n_train + self.n_validate:,:]
        self.f_train = self.f_data[:self.n_train]
        self.f_validate = self.f_data[self.n_train: self.n_train + self.n_validate]
        self.f_test = self.f_data[self.n_train + self.n_validate:]

    def prepare_data(self, filename, deg):
        self.read_data(filename)
        self.create_design_matrix(deg)
        self.split_data()


    def SGD(self, epochs, batch_size, eta, gamma=0):
        """
        Perform stochastic gradient descent (SGD), with and without momemtum.

        Input parameters
        -----------------
        epochs : (int) number of epochs
        batch_size : (int) size of each mini batch
        eta : (float) learning rate
        gamma : (float) momemtum term. Default is zero, meaning that we only perform classical SGD.

        """


        self.w = np.random.normal(size=[self.p])/self.p
        batches = self.n_train//batch_size
        total_indices = np.arange(self.n_train)

        cost_function = np.zeros(epochs+1)
        cost_function[0] = np.linalg.norm(self.f_train - self.X_train@self.w)**2

        prev_average = np.zeros(self.p)

        bar = Bar("Epoch ", max = epochs)

        for epoch in range(epochs):
            bar.next()
            for b in range(batches):
                indices = np.random.choice(total_indices, size=batch_size,replace=True)
                X = self.X_train[indices]
                z = self.f_train[indices]
                grad_cost = (-(X.T @ z) + X.T @ X @ self.w) + self.Lambda*self.w
                self.w -= (gamma*prev_average + eta*grad_cost)
                prev_average = (gamma*prev_average + eta*grad_cost)
            cost_function[1+epoch] = np.linalg.norm(self.f_train - self.X_train@self.w)**2
        bar.finish()

        return cost_function

        # Må sette inn predict shit på validering og så predict på test

    def predict_validate(self):
        f_validate = self.X_validate @ self.w
        MSE = self.compute_MSE(self.f_validate, f_validate)
        R2 = self.compute_R2_score(self.f_validate, f_validate)
        return MSE, R2

    def predict_test(self):
        f_test = self.X_test @ self.w
        MSE = self.compute_MSE(self.f_test, f_test)
        R2 = self.compute_R2_score(self.f_test, f_test)
        return MSE, R2

    def compute_MSE(self, f_data, f_model):
        """
        Computes the mean squared error between the model and the datapoints sent in

        Parameters:

        f_data - response data z(x,y)
        f_model - predicted response data.
        """
        return np.mean((f_model - f_data)**2)


    def compute_R2_score(self, f_data, f_model):
        """
        Computes the R2 score between the model and the datapoints sent in

        Parameters:

        f_data - response data z(x,y)
        f_model - predicted response data.
        """
        return 1 - np.sum((f_data - f_model) ** 2) / np.sum((f_data - np.mean(f_data)) ** 2)

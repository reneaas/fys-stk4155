import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1001)


class Regression:
    def __init__(self):
        """
        For now this is an empty useless init function
        """
        None

    def read_data(self,filename):
        """
        -------------------------------------------------
        Extracts:
        x, y: n data tuples (x,y).
        f_data: n function values f(x,y)
        self.n: Number of datapoints n
        self.p: Number of features of the model
        -------------------------------------------------
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
        -------------------------------------------------------------
        deg: degree of polynomial p(x,y)
        -------------------------------------------------------------
        """
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

    def split_data(self):
        """
        Splits the data into a training set
        Training/test is by default 80/20 ratio.
        """
        #Split data into training and test set.
        self.n_train = 4*(self.n // 5) + 4*(self.n % 5)
        self.n_test = (self.n // 5) + (self.n % 5)
        self.X_train = self.design_matrix[:self.n_train,:]
        self.X_test = self.design_matrix[self.n_train:,:]
        self.f_train = self.f_data[:self.n_train]
        self.f_test = self.f_data[self.n_train:]


    def compute_MSE(self, f_data, f_model):
        return np.mean((f_model - f_data)**2)


    def compute_R2_score(self, f_data, f_model):
        return 1 - np.sum((f_data - f_model) ** 2) / np.sum((f_data - np.mean(f_data)) ** 2)

    def compute_bias_variance(self):
        f_model = self.X_test @ self.w
        mean_model = np.mean(f_model)
        bias = np.mean((self.f_test - mean_model)**2)
        variance = np.mean((f_model-mean_model)**2)
        return bias, variance


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


    def train(self, X_train, f_train):
        return None

    def bootstrap(self, B):
        #Copy data
        X_train = np.copy(self.X_train)
        f_train = np.copy(self.f_train)

        #Run bootstrap resampling
        self.w_boots = np.zeros((B, self.p))
        for i in range(B):
            idx = np.random.randint(0,self.n_train, size=self.n_train)
            self.X_train[idx,:]
            self.f_train[idx]
            self.train()
            self.w_boots[i, :] = self.w[:]
        self.compute_statistics(self.w_boots)

        #Recopy the initial dataset.
        self.X_train = X_train
        self.f_train = f_train

    def k_fold_cross_validation(self,k):
        #Copy data
        X_train = np.copy(self.X_train)
        f_train = np.copy(self.f_train)


        int_size = self.n_train // k
        remainder = self.n_train % k
        fold_size = np.zeros(k, dtype ="int")
        for i in range(k):
            fold_size[i] = int_size + (remainder > 0)
            remainder -= 1

        #Construct row pointer
        row_ptr = np.zeros(k+1, dtype="int")
        for i in range(1,k+1):
            row_ptr[i] += row_ptr[i-1]+fold_size[i-1]

        #Perform k-fold cross validation
        self.w_k_fold = np.zeros((k, self.p))
        for j in range(k):
            X_test = self.X_train[[i for i in range(row_ptr[j],row_ptr[j+1])], :]
            f_test = self.f_train[[i for i in range(row_ptr[j],row_ptr[j+1])]]
            idx = []
            for l in range(j):
                idx += [i for i in range(row_ptr[l],row_ptr[l+1])]
            for l in range(j+1,k):
                idx += [i for i in range(row_ptr[l],row_ptr[l+1])]
            X_train = self.X_train[idx, :]
            f_train = self.f_train[idx]
            self.train(X_train, f_train)
            self.w_k_fold[j, :] = self.w[:]

        self.compute_statistics(self.w_k_fold)

        #Recopy the initial dataset.
        self.X_train = X_train
        self.f_train = f_train

    def compute_statistics(self,w):
        self.w_mean = np.zeros(self.p)
        self.w_std = np.zeros(self.p)
        for i in range(self.p):
            self.w_mean[i] = np.mean(w[:, i])
            self.w_std[i] = np.std(w[:,i])
        self.w[:] = self.w_mean[:]

    def predict_train(self):
        f_model = self.X_train @ self.w
        R2_score = self.compute_R2_score(self.f_train, f_model)
        MSE = self.compute_MSE(self.f_train, f_model)
        return R2_score, MSE

    def predict_test(self):
        f_model = self.X_test @ self.w
        R2_score = self.compute_R2_score(self.f_test, f_model)
        MSE = self.compute_MSE(self.f_test, f_model)
        return R2_score, MSE

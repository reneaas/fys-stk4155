import numpy as np
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

        p = int((deg+1) * (deg+2) / 2 - 1)

        x = np.array(x)
        y = np.array(y)
        self.func_vals = np.array(func_vals)
        self.design_matrix = np.zeros([self.n,p])

        deg = deg + 1
        index = 0

        for i in range(deg):      #Itererer over alle x^0, x^1,...., x^4
            for j in range(deg - i):      #Vil for hver x^i ha alle y opp til y^(order-i) for å få alle komboer mindre enn order grad
                if not (i==0 and j==0):     #Vil ikke ha x^0 * y^0 som en kolonne
                    self.design_matrix[:,index] = x[:]**i * y[:]**j
                    index +=1       #Må inkrementere kolonnen i matrisen ett hakk hver gang

    def LinearRegression(self):

        X_train, X_test, f_train, f_test = train_test_split(self.design_matrix, self.func_vals, test_size=0.2)

        beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ f_train
        print(beta)

        f_tilde = X_train @ beta
        f_predict = X_test @ beta

        return f_train, f_test, f_tilde, f_predict

    def R2(self, f_data, f_model):
        return 1 - np.sum((f_data - f_model) ** 2) / np.sum((f_data - np.mean(f_data)) ** 2)

    def MSE(self, f_data, f_model):
        return np.sum((f_data-f_model)**2)/self.n

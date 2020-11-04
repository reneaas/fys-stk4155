import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, SGDClassifier
from regression import Regression, OLS, Ridge
import pandas as pd
np.random.seed(1001)

filename = "datasets/frankefunction_dataset_N_10000_sigma_0.1.txt"

"""
# Constructing the validation dataset
validation_file = "datasets/frankefunction_dataset_N_200_sigma_0.1.txt"

f_validation = []
with open(validation_file, "r") as infile:
    lines = infile.readlines()
    for line in lines:
        values = line.split()
        f_validation.append(float(values[2]))

f_validation = np.array(f_validation)
f_validation = (f_validation - np.mean(f_validation))/np.std(f_validation) # Scaling the data
"""
N = 10000
polynomials = np.arange(1,6)
num_epochs = np.arange(0,24,5)[1:]
num_batches = np.arange(10,109,10)[1:]

eta = [10**(-i) for i in range(3,9)]
gamma = [10**(-i) for i in range(1,10)]

"""
deg = 5
epochs = 200
eta = 10**(-3)
gamma = 0.
batch_size = 10

"""

"""
MSE = []
R2 = []
eta = 0.0001
gamma = 0
for i in range(len(polynomials)):
    print("Polynomial degree = ", polynomials[i])
    MSE.append([])
    R2.append([])
    for b in range(len(num_batches)):
        batch_size = int(N/num_batches[b])
        for e in range(len(num_epochs)):
            my_solver = OLS()                                    # Initiates the solver.
            my_solver.read_data(filename)                       # Reads data and scales it according to Z-score
            my_solver.create_design_matrix(polynomials[i])     # Creates design matrix
            my_solver.split_data()                               # Splits the data in 20/80 test/training ratio
            my_solver.SDG(num_epochs[e],batch_size,eta,gamma)
            r2, mse = my_solver.predict_test()                       # Computes R2-score and MSE on the test data.
            MSE[i].append(mse)
            R2[i].append(r2)

MSE_dict = {}
R2_dict = {}

MSE_dict["eta"] = eta
R2_dict["eta"] = eta
for i in range(len(MSE)):
    MSE_dict["{}".format(polynomials[i])] = MSE[i]
    R2_dict["{}".format(polynomials[i])] = R2[i]

MSE_data = pd.DataFrame.from_dict(MSE_dict)
R2_data = pd.DataFrame.from_dict(R2_dict)

MSE_data.to_csv("MSE_SGD.csv")
R2_data.to_csv("R2_SGD.csv")
"""

filename = "MSE_SGD.csv"
MSE_5 = []

with open(filename, "r") as infile:
    infile.readline()
    lines = infile.readlines()
    for line in lines:
        values = line.split(",")
        MSE_5.append(float(values[-1]))

B, E = np.meshgrid(num_batches, num_epochs)
MSE_mat = np.zeros([len(num_batches), len(num_epochs)])
MSE_mat.flat[:] = MSE_5
plt.contourf(B, E, MSE_mat.T)
plt.colorbar()
plt.show()





"""
print("-------------------------------------")


sgdreg = SGDRegressor(learning_rate = "constant", max_iter = 50, penalty=None, eta0=eta)
data_indices = np.arange(int(N*0.8))

batches = int(my_solver.n_train/batch_size)

for i in range(epochs):
    indices = np.random.choice(data_indices, size = batch_size, replace=True)
    X_train = my_solver.X_train[indices]
    f_train = my_solver.f_train[indices]
    for b in range(batches):
        sgdreg.partial_fit(X_train, f_train)


#sgdreg.fit(my_solver.X_train,my_solver.f_train)

y_predict = sgdreg.predict(my_solver.X_test)
R2 = sgdreg.score(my_solver.X_test, my_solver.f_test.ravel())
MSE = my_solver.compute_MSE(my_solver.f_test, y_predict)

print("Scikit learn; R2 score = ", R2)
print("Scikit learn; MSE = ", MSE)

"""
"""
# Show-case usage of Ridge class:
my_solver = Ridge()                    # Initiates the solver.
my_solver.Lambda = 0.00001             # Regularization parameter
my_solver.read_data(filename)          # Reads data and scales it according to Z-score
my_solver.create_design_matrix(deg)    # Creates design matrix
my_solver.split_data()                 # Splits the data in 20/80 test/training ratio
#my_solver.train()                      # Computes the parameters of the model
cost_func = my_solver.SDG(epochs=epochs,batch_size=5,eta=0.0001, gamma=0.2)
R2, MSE = my_solver.predict_test()     # Computes R2-score and MSE on the test data.
print("Ridge; R2 score = ", R2)
print("Ridge; MSE = ", MSE)
"""

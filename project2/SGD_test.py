import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from regression import Regression, OLS, Ridge

np.random.seed(1001)

filename = "datasets/frankefunction_dataset_N_1000_sigma_0.1.txt"

deg = 7
epochs = 500
batch_size = 10
eta = 0.0001
gamma = 0.2

# Show-case usage of OLS class:
my_solver = OLS()                       # Initiates the solver.
my_solver.read_data(filename)           # Reads data and scales it according to Z-score
my_solver.create_design_matrix(deg)     # Creates design matrix
my_solver.split_data()                  # Splits the data in 20/80 test/training ratio
cost_func = my_solver.SDG(epochs,batch_size,eta,gamma)
R2, MSE = my_solver.predict_test()      # Computes R2-score and MSE on the test data.
print("OLS; R2 score = ", R2)
print("OLS; MSE = ", MSE)

"""
sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=eta)
sgdreg.fit(my_solver.X_train,my_solver.f_train)

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

from regression import OLS, Ridge, Lasso
import os

filename = "./datasets/frankefunction_dataset_N_1000_sigma_0.1.txt" #Test file

if not os.path.exists(filename):
    os.system("python3 generate_data.py 1000 0.1")

deg = 5 #Highest polynomial degree, gives 15 features in the model.

# Show-case usage of OLS class:
my_solver = OLS()                    # Initiates the solver.
my_solver.read_data(filename)        # Reads data and scales it according to Z-score
my_solver.create_design_matrix(deg)  # Creates design matrix
my_solver.split_data()               # Splits the data in 20/80 test/training ratio
my_solver.train()                    # Computes the parameters of the model
R2, MSE = my_solver.predict_test()   # Computes R2-score and MSE on the test data.
print("OLS; R2 score = ", R2)
print("OLS; MSE = ", MSE)

# Show-case usage of Ridge class:
my_solver = Ridge()                    # Initiates the solver.
my_solver.Lambda = 0.00001             # Regularization parameter
my_solver.read_data(filename)          # Reads data and scales it according to Z-score
my_solver.create_design_matrix(deg)    # Creates design matrix
my_solver.split_data()                 # Splits the data in 20/80 test/training ratio
my_solver.train()                      # Computes the parameters of the model
R2, MSE = my_solver.predict_test()     # Computes R2-score and MSE on the test data.
print("Ridge; R2 score = ", R2)
print("Ridge; MSE = ", MSE)

# Show-case usage of Lasso class:
my_solver = Lasso()                     # Initiates the solver.
my_solver.Lambda = 0.00001              # Regularization parameter
my_solver.read_data(filename)           # Reads data and scales it according to Z-score
my_solver.create_design_matrix(deg)     # Creates design matrix
my_solver.split_data()                  # Splits the data in 20/80 test/training ratio
my_solver.train()                       # Computes the parameters of the model
R2, MSE = my_solver.predict_test()      # Computes R2-score and MSE on the test data.
print("Lasso; R2 score = ", R2)
print("Lasso; MSE = ", MSE)


# Perform bootstrap using Ridge
B = 100                                 # Number of bootstrap samples
my_solver = Ridge()                     # Initiates the solver.
my_solver.Lambda = 0.00001              # regularization parameter
my_solver.read_data(filename)           # Reads data and scales it according to Z-score
my_solver.create_design_matrix(deg)     # Creates design matrix
my_solver.split_data()                  # Splits the data in 20/80 test/training ratio
R2, MSE, bias, variance = my_solver.bootstrap(B) # Computes R2-score, MSE, bias and variance on the test data through bootstrapping
print("Bootstrap Ridge; R2 = ", R2)
print("Bootstrap Ridge; MSE = ", MSE)
print("Bootstrap Ridge; Bias = ", bias)
print("Bootstrap Ridge; Variance = ", variance)

# Perform k-fold validation using Lasso
k = 10
my_solver = Lasso()                     # Initiates the solver.
my_solver.Lambda = 0.00001              # Regularization parameter
my_solver.read_data(filename)           # Reads data and scales it according to Z-score
my_solver.create_design_matrix(deg)     # Creates design matrix
my_solver.split_data()                  # Splits the data in 20/80 test/training ratio
R2, MSE = my_solver.k_fold_cross_validation(B) # Computes R2-score and MSE through k-fold cross-validation
print("10-fold cross-validation Ridge; R2 = ", R2)
print("10-fold cross-validation Ridge; MSE = ", MSE)

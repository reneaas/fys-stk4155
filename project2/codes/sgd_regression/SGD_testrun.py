from SGD_regression import SGD_Regression



filename = "datasets/frankefunction_dataset_N_20000_sigma_0.1.txt"

# Declare hyperparamters here

deg = 6
eta = 1e-4
epochs = 1000
batch_size = 20
gamma = 0.2
Lambda = 1e-7

# Perform stochastic gradient descent for OLS without momentum

my_solver = SGD_Regression()
my_solver.prepare_data(filename, deg)
cost_function = my_solver.SGD(epochs = epochs, bacth_size = batch_size, eta = eta)

# Perform stochastic gradient descent for OLS with momentum

my_solver = SGD_Regression()
my_solver.prepare_data(filename, deg)
cost_function = my_solver.SGD(epochs = epochs, bacth_size = batch_size, eta = eta, gamma = gamma)

# Perform stochastic gradient descent for Ridge Regression without momentum

my_solver = SGD_Regression(Lambda)
my_solver.prepare_data(filename, deg)
cost_function = my_solver.SGD(epochs = epochs, bacth_size = batch_sizes, eta = eta)

# Perform stochastic gradient descent for Ridge Regression with momentum

my_solver = SGD_Regression(Lambda)
my_solver.prepare_data(filename, deg)
cost_function = my_solver.SGD(epochs = epochs, bacth_size = batch_size, eta = eta, gamma = gamma)

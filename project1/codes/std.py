from regression import OLS, Ridge, Lasso

path = "./datasets/"
filename = path + "frankefunction_dataset_N_1000_sigma_0.1_TEST.txt"



my_solver = Lasso()
my_solver.Lambda = 10e-4
my_solver.read_data(filename)
my_solver.create_design_matrix(10)
my_solver.split_data()
R2, MSE, bias, variance = my_solver.bootstrap(100)

print("MSE = ", MSE)
print("R2 = ", R2)

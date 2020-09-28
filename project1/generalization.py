from ols import OLS
import numpy as np
import matplotlib.pyplot as plt

deg = 7
filename = "./datasets/frankefunction_dataset_N_10000_sigma_0.1.txt"

my_solver = OLS()
my_solver.read_data(filename)
my_solver.create_design_matrix(deg)
my_solver.split_data()
my_solver.train()

filename_terrain = "./datasets/TerrainFiles/terrain_data.txt"
my_solver.read_data(filename_terrain)
my_solver.create_design_matrix(deg)
my_solver.split_data()
R2, MSE = my_solver.predict_test()

print("R2 = ", R2)

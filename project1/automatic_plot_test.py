import sys
import os

N = [100,1000]
sigma = [0.1, 1.0]
deg = [5,8,15,20]

for n in N:
    for s in sigma:
        for p in deg:
            os.system("python3 plot_bootstrap_kfold.py" + " " + str(n) + " " + str(s) + " " + str(p))

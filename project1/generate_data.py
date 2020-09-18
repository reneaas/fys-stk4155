import numpy as np
import sys
import os

N = int(sys.argv[1])
sigma = float(sys.argv[2])

x = np.random.uniform(0,1, N)
y = np.random.uniform(0,1, N)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def f(x,y):
    """
    Test function to check that the Regression class produces correct results.
    """
    return  1 + 2*x + 4*y + x**2 + 2*x*y + + y**2

noise = np.random.normal(0,sigma,size=N)

data = FrankeFunction(x,y) + noise
#data = f(x,y)

names = ["frankefunction", "dataset", "N", str(N), "sigma", str(sigma)]
outfilename = "_".join(names) + ".txt"

with open(outfilename, "w") as outfile:
    for i in range(N):
        outfile.write(str(x[i]) + " " + str(y[i]) + " " + str(data[i]))
        outfile.write("\n")

path = "datasets"
if not os.path.exists(path):
    os.makedirs(path)

os.system("mv" + " " + outfilename + " " + path)

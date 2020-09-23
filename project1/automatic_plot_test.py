import sys
import os
from progress.bar import Bar

N = [100,1000]
sigma = [0.1, 1.0]
deg = [5,8,15,20]

path_to_datasets = "./datasets/"
for n in N:
    for s in sigma:
        filename = path_to_datasets + "_".join(["frankefunction", "dataset", "N", str(n), "sigma", str(s)]) + ".txt"
        if not os.path.exists(filename):
            os.system(" ".join(["python3", "generate_data.py", str(n), str(s)]))



for n in N:
    print("N = ", n)
    for s in sigma:
        bar = Bar("Progress", max = len(deg))
        for p in deg:
            os.system("python3 plot_bootstrap_kfold.py" + " " + str(n) + " " + str(s) + " " + str(p))
            bar.next()
        bar.finish()

import numpy as np
import os
import sys



path = "results/euler"
if not os.path.exists(path):
    os.makedirs(path)

dx = [0.1, 0.01]
total_time = [0.02, 1.]
r = 0.5

for i in dx:
    for t in total_time:
        outfilename = "euler_dx_" + str(i) + "_time_" + str(t) + ".txt"
        command = " ".join(["./main.exe", outfilename, str(t), str(i)])
        os.system(command)
        os.system("mv" + " " + outfilename +" "+ path)

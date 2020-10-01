import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
plt.rc("text", usetex=True)


# Load the terrain
terrain1 = imread("./datasets/TerrainFiles/SRTM_data_Norway_1.tif")
# Show the terrain


n = len(terrain1)
m = len(terrain1[0])

path_to_plot = "./results/TerrainData/IMGS/"
if not os.path.exists(path_to_plot):
    os.makedirs(path_to_plot)

plot_name = path_to_plot + "terrain_image_2.pdf"

terrain1 = terrain1[::-1,:]
print(terrain1)
N_x = 1000
N_y = 1000
start_x = 0
start_y = 0
x = np.linspace(0, m-1, m)
y = np.linspace(0, n-1, n)
X,Y = np.meshgrid(x,y)
X = X[start_x:start_x + N_x, start_x:start_x + N_x]
Y = Y[start_y:start_y + N_y,start_y: start_y + N_y]
Z = terrain1[start_x:start_x + N_x, start_y:start_y + N_y]
#Z = terrain1

font_size = 16
tick_size = 14
plt.figure()
plt.imshow(Z, cmap="gray")
plt.xlabel(r"$x$", fontsize=font_size)
plt.ylabel(r"$y$", fontsize=font_size)
plt.xticks(size=tick_size)
plt.yticks(size=tick_size)
cb = plt.colorbar()
cb.set_label(label="$z(x,y)$", size=16)
cb.ax.tick_params(labelsize=14)
#plt.savefig(plot_name)
plt.show()


x_data = X.flat[:]
y_data = Y.flat[:]
z_data = Z.flat[:]


"""
outfilename = "./datasets/TerrainFiles/terrain_data_2.txt"

with open(outfilename, "w") as outfile:
    for i in range(len(x_data)):
        outfile.write(" ".join([str(x_data[i]), str(y_data[i]), str(z_data[i])]))
        outfile.write("\n")
"""

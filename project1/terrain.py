import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread("./datasets/DataFiles/SRTM_data_Norway_2.tif")
# Show the terrain

print(type(terrain1), len(terrain1))
plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


n = len(terrain1)
m = len(terrain1[0])
print(n,m)

print(np.size(terrain1))
terrain1 = terrain1[::-1,:]

x = np.linspace(0, m-1, m)
y = np.linspace(0, n-1, n)
X,Y = np.meshgrid(x,y)


print("X = ", X)
print("Y = ", Y)
plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
Z = np.zeros((n,m))
Z.flat[:] = terrain1[:]
print("Z = ", Z)

x_data = np.zeros(n*m)
y_data = np.zeros(n*m)
z_data = np.zeros(n*m)



x_data[:] = X.flat[:]
y_data[:] = Y.flat[:]
z_data[:] = Z.flat[:]


plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.axis("equal")
plt.figure()
plt.contourf(X,Y,Z, cmap="gray")
plt.colorbar()
plt.show()

outfilename = "terrain_data.txt"

with open(outfilename, "w") as outfile:
    for i in range(len(x_data)):
        outfile.write(" ".join([str(x_data[i]), str(y_data[i]), str(z_data[i])]))
        outfile.write("\n")

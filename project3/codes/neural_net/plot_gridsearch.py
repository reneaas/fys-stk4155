import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

loaded_mat = np.load('grid_search_R2_pde.npy')
num_layers = [2,4,6,8,10]
num_nodes = [10,50,100,500,1000]

sb.set(font_scale=1.25)
heat_map = sb.heatmap(loaded_mat.T, annot=True, cbar=True, cbar_kws={"label": "$R^2$", "orientation" : "vertical"})
heat_map.set_xlabel("Hidden Layers")
heat_map.set_ylabel("Nodes")
heat_map.set_xticklabels(num_layers)
heat_map.set_yticklabels(num_nodes)
heat_map.xaxis.tick_top()
#print(heat_map.get_ylim())
heat_map.set_ylim(5.0,0.0)
heat_map.tick_params(length=0)
plt.savefig("../results/neural_net/pde/r2_grid_search.pdf")
plt.show()

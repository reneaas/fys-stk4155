# Neural Nets

This README explains how to use the neural network solvers created for this project. To be able to use these solvers, make sure you have the required packages *tensorflow*, *numpy*, *matplotlib* and *progress*. If not, you can install them with *pip* using

```command
pip3 install tensorflow numpy matplotlib progress
```


## The Neural Diffusion Solver

The neural diffusion solver is found in *neural_diffusion_solver.py*. It's designed to be able to solve any 1D diffusion equation for which a trial function must be provided. In the following we illustrate its nuts and bolts, and how to use them.


#### Constructor

```Python
from neural_diffusion_solver import NeuralDiffusionSolver

my_solver = NeuralDiffusionSolver(layers=layers, input_sz=input_sz)
```


| Args    |     |
| :------------- | :------------- |
| layers   |  list contaning number of hidden layers (including output layer). layers = [hidden_layer1, ..., hidden_layerN, output_layer]. |
| input_sz | size of inputs. |



#### Solving the PDE

At this point, we can solve the PDE. This is easily done by the following code:

```Python
x = np.random.uniform(0, 1, number_of_points)
t = np.random.uniform(0, t_max, number_of_points)
epoch_arr, loss = my_model.fit(x=x, t=t, epochs=epochs) #Returns loss as a function of epochs.
```


#### Computing predictions
Once the model is fitted, we can evaluate the function at any time 0 <= x <= 1 and any t > 0.

```Python
prediction = my_solver.predict(x, t)
```

where x and t are tensors dtype=tf.float32.

# Neural Network Solvers for Differential Equations

This README explains how to use the neural network solvers created for this project. To be able to use these solvers, make sure you have the required packages *tensorflow*, *numpy*, *matplotlib* and *progress*. If not, you can install them with *pip* using

```command
pip3 install tensorflow numpy matplotlib progress
```

Both classes here inherit from the base class *neural_base.py*.

## The Neural Diffusion Solver

The neural diffusion solver is found in *neural_diffusion_solver.py*. It's designed to be able to solve any 1D diffusion equation with Dirichlet boundary conditions set to 0. In the following we illustrate its nuts and bolts, and how to use them.


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

#### test program
In *test_diffusion.py*, an end-to-end example is provided where the model of the neural diffusion solver is fitted and tested. Simply run

```terminal
python3 test_diffusion.py
```


## The Neural Eigensolver

The neural eigensolver is found in *neural_eigensolver.py*. It's capable of computing eigenvectors and eigenvalues of symmetric matrices.

#### Constructor

```Python
from neural_eigensolver import NeuralEigenSolver

my_solver = NeuralEigenSolver(layers = layers, input_sz = input_sz, matrix = A, eig_type = eig_type)
```

| Args    |     |
| :------------- | :------------- |
| layers   |  list contaning number of hidden layers (including output layer). layers = [hidden_layer1, ..., hidden_layerN, output_layer]. |
| input_sz | size of inputs. |
| matrix | A symmetric matrix, 2D numpy array. |
| eig_type | Either "max" or "min". Will in principle return the maximum or minimum eigenvalue, but discrepancies occur. Convergence to some eigenvector is typically guaranteed. |

#### Computing an eigenvalue and an eigenvector

The network converges to *some* eigenvector of the matrix A. In the following we show how to fit the network and compute a normalized eigenvector and its corresponding eigenvalue

```Python
#Fit the model
Nt = 50 #Number of timesteps
t_max = 1e3 #Maximum value for t.
x = np.random.normal(0, 1, size=mat_sz) #Random initial vector x(0).
t = np.linspace(0, t_max, Nt)
epochs = 2500
epoch_arr, eigvals, eigvecs = my_solver.fit(x = x, t = t, epochs = epochs)

#Compute eigenvalue and normalized eigenvector
eigval, eigvec = my_solver.eig(x, t)
```

The variables returned by the *fit* method keeps track of estimates of the eigenvector and eigenvalue as a function of number of epochs.

#### test program
In *test_eig.py*, an end-to-end example is provided where a model of the neural eigensolver is fitted and tested. Simply run

```terminal
python3 test_eig.py
```

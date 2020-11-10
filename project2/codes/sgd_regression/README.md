# Stochastic gradient descent (SGD)

We have implemented two different methods of stochastic gradient descent:
1. SGD with mini-batches
2. SGD with mini-batches and momentum



## Code usage

The python script *SGD_testrun.py* describes how to use the class SGD_Regression through examples for different scenarios. To perform SGD either with or without momentum requires only three step:
 1. Initialize your solver, either with or without L2 regularization.
 2. Prepare the data through the class method *prepare_data*, which takes the datafile and polynomial degree as input.
 3. Finally call the *SGD* method. This method takes the following input parameters: epochs, batch size, learning rate, momentum term.

For a more thorough description of the different class methods, see the doc string documentation in *SGD_regression.py*.

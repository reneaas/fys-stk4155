# Project 2 - Neural Network is tha shit

This project take a deep dive into masse shit om hva vi gjør. Kan skrive dette etter vi har skrevet intro.

The written report can be found [here](https://github.com/reneaas/fys-stk4155/tree/master/project2/report).

## Codes

To run the codes of the project, we advise to clone the repository pertaining to this project.

Below follows a short description of the different codes used to solve the encountered problems in this project.

1. SGD_regression.py for performing stochastic gradient descent
   * A class which preprocess the data, and minimizes the mean square error for ordinary least squares and Ridge regression.
2. Resten av kodene som jeg ikke aner hva heter eller gjør


## Generate data

how???

## Stochastic gradient descent (SGD)

We have implemented two different methods of stochastic gradient descent:
1. SGD with mini-batches
2. SGD with mini-batches and momentum



### Code usage

The python script *SGD_testrun.py* describes how to use the class SGD_Regression through examples for different scenarios. To perform SGD either with or without momentum requires only three step:
 1. Initialize your solver, either with or without L2 regularization.
 2. Prepare the data through the class method *prepare_data*, which takes the datafile and polynomial degree as input.
 3. Finally call the *SGD* method. This method takes the following input parameters: epochs, batch size, learning rate, momentum term.

For a more thorough description of the different class methods, see the doc string documentation in *SGD_regression.py*.

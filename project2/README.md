# Project 2 - Neural Network is tha shit

This project take a deep dive into masse shit om hva vi gjør. Kan skrive dette etter vi har skrevet intro.

The written report can be found [here](https://github.com/reneaas/fys-stk4155/tree/master/project2/report).

## Codes

To run the codes of the project, we advise to clone the repository pertaining to this project.

Below follows a short description of the different codes used to solve the encountered problems in this project.

1. SGD_regression.py for performing stochastic gradient descent
   * A class which preprocess the data, and minimizes the mean square error for ordinary least squares and Ridge regression.
2. Resten av kodene som jeg ikke aner hva heter eller gjør


## Stochastic gradient descent (SGD)

We have implemented two different methods of stochastic gradient descent:
1. SGD with mini-batches
2. SGD with mini-batches and momentum

### Producing data

We perform SGD on data from Frankes function, which can be produced by running the script *generate_data.py* with the following in a Linux/Unix command line:

```console
python3 generate_data.py N sigma
```

* N : Number of datapoints
* sigma : The desired standard deviation of the noise added to the data

### Code usage

The python script *SGD_testrun.py* shows how to use the class SGD_Regression for the two SGD methods. For a more thorough description of the different class methods, see the doc string documentation in *SGD_regression.py*.
